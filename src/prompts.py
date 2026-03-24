# Intended pipeline:
# 1. Send a Vulnerability Detector (VD) prompt (one of the PROMPT_VD*) with {input} replaced by the Solidity file.
# 2. Send the model’s free-form answer to a SA prompt to get standardized lines like: Reentrancy: 0, Access Control: 1...

ROLE_VD = "You are a vulnerability detector for a smart contract. "
ROLE_SA = "You are a semantic analyzer of text. "

COT = "Reason carefully and base each conclusion on the contract code. "

VULNS = """
ID: Access Control
Description: Improper restriction of function access, allowing unauthorized users to execute critical functions.
         
ID: Arithmetic
Description: Missing or incorrect integer overflow/underflow checks, leading to unexpected values.

ID: Bad Randomness
Description: Use of predictable values for randomness, which attackers can manipulate.

ID: Denial Of Service
Description: Malicious actions that consume excessive gas or exploit fallback behavior to block contract functionality.

ID: Front Running
Description: Exploiting transaction order to manipulate outcomes before others.

ID: Reentrancy
Description: Recursive calls into the same contract before the initial function completes, leading to unexpected state changes.

ID: Short Addresses
Description: Failure to validate input lengths, allowing crafted shorter inputs to corrupt parameters.

ID: Time Manipulation
Description: Dependence on block timestamps that miners can influence.

ID: Unchecked Low Level Calls
Description: Calls to low-level functions like 'call()' without checking return values, which can silently fail.
"""


FEW_SHOT = """
- Example 1
Input:
```
function initContract() public {
	owner = msg.sender;
}
```
Output: The contract's initialization function sets the caller as the owner but is not part of the constructor. It lacks access restrictions and does not track whether it has already been called. As a result, any user can invoke the function and assign ownership to themselves, potentially leading to unauthorized access to privileged functionality. Verdict, Access Control is present.


- Example 2
Input:
```
function withdraw(uint _amount) {
	require(balances[msg.sender] - _amount > 0);
	msg.sender.transfer(_amount);
	balances[msg.sender] -= _amount;
}
```
Output: The function does not check for integer underflow, allowing you to withdraw an infinite amount of tokens. Verdict, Arithmetic is present.


- Example 3
Input:
```
uint256 private seed;

function play() public payable {
	require(msg.value >= 1 ether);
	if (block.blockhash(blockNumber) % 2 == 0) {
		msg.sender.transfer(this.balance);
	}
}
```
Output: A private seed is used in combination with an iteration number and the keccak256 hash function to determine if the caller wins. Even though the seed is private, it must have been set via a transaction at some point in time and thus is visible on the blockchain. Verdict, Bad Randomness is present.


- Example 4
Input:
```
function becomePresident() payable {
    require(msg.value >= price); 
    president.transfer(price);   
    president = msg.sender;      
    price = price * 2;           
}
```
Output: The function allows you to become the president if you publicly bribe the previous one. Unfortunately, if the previous president is a smart contract and causes reversion on payment, the transfer of power will fail and the malicious smart contract will remain president forever. Verdict, Denial Of Service is present.


- Example 5
Input:
```
function solve(string memory solution) public {
    require(
        hash == keccak256(abi.encodePacked(solution)), "Incorrect answer"
    );
    (bool sent,) = msg.sender.call{value : 10 ether}("");
    require(sent, "Failed to send Ether");
}
```
Output: The function leaks the secret solution in the transaction data, allowing an attacker to front-run the original user by submitting the same solution with a higher gas fee, thereby stealing the Ether reward. Verdict, Front Running is present.


- Example 6
Input:
```
function withdraw(uint _amount) {
	require(balances[msg.sender] >= _amount);
	msg.sender.call.value(_amount)();
	balances[msg.sender] -= _amount;
}
```
Output: The function is vulnerable to a reentrancy attack. When the low level call() function sends ether to the msg.sender address, it becomes vulnerable; if the address is a smart contract, the payment will trigger its fallback function with what's left of the transaction gas. Verdict, Reentrancy is present.


- Example 7
Input:
```
function transfer(address _to, uint256 _value) public {
    require(balances[msg.sender] >= _value);
    balances[msg.sender] -= _value;
    balances[_to] += _value;
    emit Transfer(msg.sender, _to, _value);
}
```
Output: An attacker can exploit the contract by sending a transaction with a malformed 19-byte address instead of the standard 20 bytes. Because the contract reads a fixed 20-byte address, it mistakenly includes the first byte of the token amount as part of the address. This misalignment results in the contract interpreting a much larger token amount—up to 256 times more—than intended, enabling the attacker to steal excess tokens. Verdict, Short Addresses is present.


- Example 8
Input:
```
function play() public {
	require(now > 1521763200 && neverPlayed == true);
	neverPlayed = false;
	msg.sender.transfer(1500 ether);
}
```
Output: The function only accepts calls that come after a specific date. Since miners can influence their block's timestamp (to a certain extent), they can attempt to mine a block containing their transaction with a block timestamp set in the future. If it is close enough, it will be accepted on the network and the transaction will give the miner ether before any other player could have attempted to win the game. Verdict, Time Manipulation is present.


- Example 9
Input:
```
function withdraw(uint256 _amount) public {
	require(balances[msg.sender] >= _amount);
	balances[msg.sender] -= _amount;
	etherLeft -= _amount;
	msg.sender.send(_amount);
}
```
Output: If the external call is used to send ether to a smart contract that does not accept them (e.g. because it does not have a payable fallback function), the EVM will replace its return value with false. Since the return value is not checked in our example, the function's changes to the contract state will not be reverted, and the etherLeft variable will end up tracking an incorrect value. Verdict, Unchecked Low Level Calls is present.
"""

VD_SUFFIX = """
Analyze the smart contract for the nine vulnerabilities listed above.

For each vulnerability:
- Output exactly one line in the format:
  <ID>: Present | Absent | Uncertain
- Then provide a brief explanation.

Rules:
- Use the exact ID strings provided.
- Do not skip any vulnerability.
- Base conclusions strictly on the code.
- Be conservative: if unsure, use "Uncertain".
"""

TASK_VD = """Here are nine common vulnerabilities:
""" + VULNS + VD_SUFFIX

TASK_VD_FEW_SHOT = """Here are nine common vulnerabilities:
""" + VULNS + """

Here are examples of how individual vulnerabilities may be identified and explained:
""" + FEW_SHOT + VD_SUFFIX

TASK_SA = """Here are nine common vulnerabilities:
""" + VULNS + """

The following text is a vulnerability detection result for a smart contract.

Convert it into a complete binary classification.

Rules:
- Output exactly nine lines, one for each vulnerability ID above.
- Use the exact IDs and the exact order shown below.
- Each line must be in the format: <ID>: <0 or 1>
- Use 1 only if the text clearly claims or strongly implies that the vulnerability is present.
- Use 0 if the text says it is absent, does not mention it, or is uncertain.
- If the text is ambiguous, prefer 0.
- Do not add explanations or extra text.

Output format:
Access Control: 0
Arithmetic: 0
Bad Randomness: 0
Denial Of Service: 0
Front Running: 0
Reentrancy: 0
Short Addresses: 0
Time Manipulation: 0
Unchecked Low Level Calls: 0
"""

INPUT = "\nThe input is:\n{input}"

################################################ PROMPTS ################################################
SA = TASK_SA + INPUT

# PERSONA variants are given through execution flags.
ZS = TASK_VD + INPUT
ZS_COT = TASK_VD + COT + INPUT
FS = TASK_VD_FEW_SHOT + INPUT


# Original vulnerability descriptions. We now use the structured descriptions.
# ORIGINAL_VULNS = """
# First, Reentrancy, also known as or related to race to empty, recursive call vulnerability, call to the unknown.
# Second, Access Control.
# Third, Arithmetic Issues, also known as integer overflow and integer underflow.
# Fourth, Unchecked Return Values For Low Level Calls, also known as or related to silent failing sends, unchecked-send.
# Fifth, Denial of Service, including gas limit reached, unexpected throw, unexpected kill, access control breached.
# Sixth, Bad Randomness, also known as nothing is secret.
# Seventh, Front-Running, also known as time-of-check vs time-of-use (TOCTOU), race condition, transaction ordering dependence (TOD).
# Eighth, Time manipulation, also known as timestamp dependence.
# Nineth, Short Address Attack, also known as or related to off-chain issues, client vulnerabilities.
# """