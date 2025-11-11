# Builds prompt templates defining:
# - The ROLE (Vulnerability Detector or Semantic Analyzer).
# - Reasoning Cue (e.g. COT = "Think step by step, carefully. ").
# - Vulnerabilities taxonomy (in ORIGINAL_VULNS, and VULNS).
# - Few Shots blocks (FEW_SHOTS_1/2/3).
# - Task Instructions for the roles.
# - Input template to inject the contract code.
# Intended pipeline:
# 1. Send a Vulnerability Detector (VD) prompt (one of the PROMPT_VD*) with {input} replaced by the Solidity file.
# 2. Send the model’s free-form answer to a SA prompt to get standardized lines like: Reentrancy: 0, Access Control: 1...

ROLE_VD = "You are a vulnerability detector for a smart contract. "
ROLE_SA = "You are a semantic analyzer of text. "

COT = "Think step by step, carefully. "
# Add vulnerability descriptions

ORIGINAL_VULNS = """
First, Reentrancy, also known as or related to race to empty, recursive call vulnerability, call to the unknown. 
Second, Access Control. 
Third, Arithmetic Issues, also known as integer overflow and integer underflow. 
Fourth, Unchecked Return Values For Low Level Calls, also known as or related to silent failing sends, unchecked-send. 
Fifth, Denial of Service, including gas limit reached, unexpected throw, unexpected kill, access control breached. 
Sixth, Bad Randomness, also known as nothing is secret. 
Seventh, Front-Running, also known as time-of-check vs time-of-use (TOCTOU), race condition, transaction ordering dependence (TOD). 
Eighth, Time manipulation, also known as timestamp dependence. 
Nineth, Short Address Attack, also known as or related to off-chain issues, client vulnerabilities. 
"""

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
Description: Calls to low-level functions like 'call()' without checking return values, which can silently fail."""

FEW_SHOTS_1 = """
ID: Access Control
Description: Improper restriction of function access, allowing unauthorized users to execute critical functions.
- Example
Input:
```
function initContract() public {
	owner = msg.sender;
}
```
Output: "Access Control: 1"

ID: Arithmetic
Description: Missing or incorrect integer overflow/underflow checks, leading to unexpected values.
- Example
Input:
```
function withdraw(uint _amount) {
	require(balances[msg.sender] - _amount > 0);
	msg.sender.transfer(_amount);
	balances[msg.sender] -= _amount;
}
```
Output: "Arithmetic: 1"

ID: Bad Randomness
Description: Use of predictable values for randomness, which attackers can manipulate.
- Example
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
Output: "Bad Randomness: 1"

ID: Denial Of Service
Description: Malicious actions that consume excessive gas or exploit fallback behavior to block contract functionality.
- Example
Input:
```
function becomePresident() payable {
    require(msg.value >= price); 
    president.transfer(price);   
    president = msg.sender;      
    price = price * 2;           
}
```
Output: "Denial Of Service: 1"

ID: Front Running
Description: Exploiting transaction order to manipulate outcomes before others.
- Example
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
Output: "Front Running: 1"


ID: Reentrancy
Description: Recursive calls into the same contract before the initial function completes, leading to unexpected state changes.
- Example
Input:
```
function withdraw(uint _amount) {
	require(balances[msg.sender] >= _amount);
	msg.sender.call.value(_amount)();
	balances[msg.sender] -= _amount;
}
```
Output: "Reentrancy: 1"


ID: Short Addresses
Description: Failure to validate input lengths, allowing crafted shorter inputs to corrupt parameters.
- Example
Input:
```
function transfer(address _to, uint256 _value) public {
    require(balances[msg.sender] >= _value);
    balances[msg.sender] -= _value;
    balances[_to] += _value;
    emit Transfer(msg.sender, _to, _value);
}
```
Output: "Short Addresses: 1"


ID: Time Manipulation
Description: Dependence on block timestamps that miners can influence.
- Example
Input:
```
function play() public {
	require(now > 1521763200 && neverPlayed == true);
	neverPlayed = false;
	msg.sender.transfer(1500 ether);
}
```
Output: "Time Manipulation: 1"


ID: Unchecked Low Level Calls
Description: Calls to low-level functions like 'call()' without checking return values, which can silently fail.
- Example
Input:
```
function withdraw(uint256 _amount) public {
	require(balances[msg.sender] >= _amount);
	balances[msg.sender] -= _amount;
	etherLeft -= _amount;
	msg.sender.send(_amount);
}
```
Output: "Unchecked Low Level Calls: 1" """

FEW_SHOTS_2 = """
ID: Access Control
Description: Improper restriction of function access, allowing unauthorized users to execute critical functions.
- Example
Input:
```
function initContract() public {
	owner = msg.sender;
}
```
Output: The contract's initialization function sets the caller as the owner but is not part of the constructor. It lacks access restrictions and does not track whether it has already been called. As a result, any user can invoke the function and assign ownership to themselves, potentially leading to unauthorized access to privileged functionality.

ID: Arithmetic
Description: Missing or incorrect integer overflow/underflow checks, leading to unexpected values.
- Example
Input:
```
function withdraw(uint _amount) {
	require(balances[msg.sender] - _amount > 0);
	msg.sender.transfer(_amount);
	balances[msg.sender] -= _amount;
}
```
Output: The function does not check for integer underflow, allowing you to withdraw an infinite amount of tokens.

ID: Bad Randomness
Description: Use of predictable values for randomness, which attackers can manipulate.
- Example
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
Output: A private seed is used in combination with an iteration number and the keccak256 hash function to determine if the caller wins. Even though the seed is private, it must have been set via a transaction at some point in time and thus is visible on the blockchain.

ID: Denial Of Service
Description: Malicious actions that consume excessive gas or exploit fallback behavior to block contract functionality.
- Example
Input:
```
function becomePresident() payable {
    require(msg.value >= price); 
    president.transfer(price);   
    president = msg.sender;      
    price = price * 2;           
}
```
Output: The function allows you to become the president if you publicly bribe the previous one. Unfortunately, if the previous president is a smart contract and causes reversion on payment, the transfer of power will fail and the malicious smart contract will remain president forever.

ID: Front Running
Description: Exploiting transaction order to manipulate outcomes before others.
- Example
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
Output: The function leaks the secret solution in the transaction data, allowing an attacker to front-run the original user by submitting the same solution with a higher gas fee, thereby stealing the Ether reward.

ID: Reentrancy
Description: Recursive calls into the same contract before the initial function completes, leading to unexpected state changes.
- Example
Input:
```
function withdraw(uint _amount) {
	require(balances[msg.sender] >= _amount);
	msg.sender.call.value(_amount)();
	balances[msg.sender] -= _amount;
}
```
Output: The function is vulnerable to a reentrancy attack. When the low level call() function sends ether to the msg.sender address, it becomes vulnerable; if the address is a smart contract, the payment will trigger its fallback function with what's left of the transaction gas.

ID: Short Addresses
Description: Failure to validate input lengths, allowing crafted shorter inputs to corrupt parameters.
- Example
Input:
```
function transfer(address _to, uint256 _value) public {
    require(balances[msg.sender] >= _value);
    balances[msg.sender] -= _value;
    balances[_to] += _value;
    emit Transfer(msg.sender, _to, _value);
}
```
Output: An attacker can exploit the contract by sending a transaction with a malformed 19-byte address instead of the standard 20 bytes. Because the contract reads a fixed 20-byte address, it mistakenly includes the first byte of the token amount as part of the address. This misalignment results in the contract interpreting a much larger token amount—up to 256 times more—than intended, enabling the attacker to steal excess tokens.

ID: Time Manipulation
Description: Dependence on block timestamps that miners can influence.
- Example
Input:
```
function play() public {
	require(now > 1521763200 && neverPlayed == true);
	neverPlayed = false;
	msg.sender.transfer(1500 ether);
}
```
Output: The function only accepts calls that come after a specific date. Since miners can influence their block's timestamp (to a certain extent), they can attempt to mine a block containing their transaction with a block timestamp set in the future. If it is close enough, it will be accepted on the network and the transaction will give the miner ether before any other player could have attempted to win the game.


ID: Unchecked Low Level Calls
Description: Calls to low-level functions like 'call()' without checking return values, which can silently fail.
- Example
Input:
```
function withdraw(uint256 _amount) public {
	require(balances[msg.sender] >= _amount);
	balances[msg.sender] -= _amount;
	etherLeft -= _amount;
	msg.sender.send(_amount);
}
```
Output: If the external call is used to send ether to a smart contract that does not accept them (e.g. because it does not have a payable fallback function), the EVM will replace its return value with false. Since the return value is not checked in our example, the function's changes to the contract state will not be reverted, and the etherLeft variable will end up tracking an incorrect value. """

FEW_SHOTS_3 = """
ID: Access Control
Description: Improper restriction of function access, allowing unauthorized users to execute critical functions.
- Example
Input:
```
function initContract() public {
	owner = msg.sender;
}
```
Output: The contract's initialization function sets the caller as the owner but is not part of the constructor. It lacks access restrictions and does not track whether it has already been called. As a result, any user can invoke the function and assign ownership to themselves, potentially leading to unauthorized access to privileged functionality. Verdict, Access Control: 1

ID: Arithmetic
Description: Missing or incorrect integer overflow/underflow checks, leading to unexpected values.
- Example
Input:
```
function withdraw(uint _amount) {
	require(balances[msg.sender] - _amount > 0);
	msg.sender.transfer(_amount);
	balances[msg.sender] -= _amount;
}
```
Output: The function does not check for integer underflow, allowing you to withdraw an infinite amount of tokens. Verdict, Arithmetic: 1

ID: Bad Randomness
Description: Use of predictable values for randomness, which attackers can manipulate.
- Example
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
Output: A private seed is used in combination with an iteration number and the keccak256 hash function to determine if the caller wins. Even though the seed is private, it must have been set via a transaction at some point in time and thus is visible on the blockchain. Verdict, Bad Randomness: 1

ID: Denial Of Service
Description: Malicious actions that consume excessive gas or exploit fallback behavior to block contract functionality.
- Example
Input:
```
function becomePresident() payable {
    require(msg.value >= price); 
    president.transfer(price);   
    president = msg.sender;      
    price = price * 2;           
}
```
Output: The function allows you to become the president if you publicly bribe the previous one. Unfortunately, if the previous president is a smart contract and causes reversion on payment, the transfer of power will fail and the malicious smart contract will remain president forever. Verdict, Denial Of Service: 1

ID: Front Running
Description: Exploiting transaction order to manipulate outcomes before others.
- Example
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
Output: The function leaks the secret solution in the transaction data, allowing an attacker to front-run the original user by submitting the same solution with a higher gas fee, thereby stealing the Ether reward. Verdict, Front Running: 1

ID: Reentrancy
Description: Recursive calls into the same contract before the initial function completes, leading to unexpected state changes.
- Example
Input:
```
function withdraw(uint _amount) {
	require(balances[msg.sender] >= _amount);
	msg.sender.call.value(_amount)();
	balances[msg.sender] -= _amount;
}
```
Output: The function is vulnerable to a reentrancy attack. When the low level call() function sends ether to the msg.sender address, it becomes vulnerable; if the address is a smart contract, the payment will trigger its fallback function with what's left of the transaction gas. Verdict, Reentrancy: 1

ID: Short Addresses
Description: Failure to validate input lengths, allowing crafted shorter inputs to corrupt parameters.
- Example
Input:
```
function transfer(address _to, uint256 _value) public {
    require(balances[msg.sender] >= _value);
    balances[msg.sender] -= _value;
    balances[_to] += _value;
    emit Transfer(msg.sender, _to, _value);
}
```
Output: An attacker can exploit the contract by sending a transaction with a malformed 19-byte address instead of the standard 20 bytes. Because the contract reads a fixed 20-byte address, it mistakenly includes the first byte of the token amount as part of the address. This misalignment results in the contract interpreting a much larger token amount—up to 256 times more—than intended, enabling the attacker to steal excess tokens. Verdict, Short Addresses: 1

ID: Time Manipulation
Description: Dependence on block timestamps that miners can influence.
- Example
Input:
```
function play() public {
	require(now > 1521763200 && neverPlayed == true);
	neverPlayed = false;
	msg.sender.transfer(1500 ether);
}
```
Output: The function only accepts calls that come after a specific date. Since miners can influence their block's timestamp (to a certain extent), they can attempt to mine a block containing their transaction with a block timestamp set in the future. If it is close enough, it will be accepted on the network and the transaction will give the miner ether before any other player could have attempted to win the game. Verdict, Time Manipulation: 1


ID: Unchecked Low Level Calls
Description: Calls to low-level functions like 'call()' without checking return values, which can silently fail.
- Example
Input:
```
function withdraw(uint256 _amount) public {
	require(balances[msg.sender] >= _amount);
	balances[msg.sender] -= _amount;
	etherLeft -= _amount;
	msg.sender.send(_amount);
}
```
Output: If the external call is used to send ether to a smart contract that does not accept them (e.g. because it does not have a payable fallback function), the EVM will replace its return value with false. Since the return value is not checked in our example, the function's changes to the contract state will not be reverted, and the etherLeft variable will end up tracking an incorrect value. Verdict, Unchecked Low Level Calls: 1 """


TASK_VD = "Here are nine common vulnerabilities: " + VULNS + "\nCheck the following smart contract for the above vulnerabilities. "
TASK_VD_RP = "Here are nine common vulnerabilities: " + ORIGINAL_VULNS + "\nCheck the following smart contract for the above vulnerabilities. "

TASK_VD_FEW_SHOTS_1 = "Here are nine common vulnerabilities: " + FEW_SHOTS_1 + "\nCheck the following smart contract for the above vulnerabilities. "
TASK_VD_FEW_SHOTS_2 = "Here are nine common vulnerabilities: " + FEW_SHOTS_2 + "\nCheck the following smart contract for the above vulnerabilities. "
TASK_VD_FEW_SHOTS_3 = "Here are nine common vulnerabilities: " + FEW_SHOTS_3 + "\nCheck the following smart contract for the above vulnerabilities. "

TASK_VD_FEW_SHOTS = "Here are nine common vulnerabilities: " + FEW_SHOTS_3 + "\nCheck the following smart contract for the above vulnerabilities. "

TASK_SA = "Here are nine common vulnerabilities. " + VULNS + "The following text is a vulnerability detection result for a smart contract. Use 0 or 1 to indicate whether there are specific types of vulnerabilities. No explanations or extra text. For example: “Reentrancy: 1”. "
TASK_SA_RP = "Here are nine common vulnerabilities: " + ORIGINAL_VULNS + "The following text is a vulnerability detection result for a smart contract. Use 0 or 1 to indicate whether there are specific types of vulnerabilities. No explanations or extra text. For example: “Reentrancy: 1”. "

INPUT = "\nThe input is:\n{input}."



################################################ PROMPTS ################################################

ORIGINAL_PROMPT_SA = ROLE_SA + TASK_SA + COT + INPUT
ORIGINAL_PROMPT_SA_RP = ROLE_SA + TASK_SA_RP + COT + INPUT

ORIGINAL_PROMPT_VD = ROLE_VD + TASK_VD + COT + INPUT
ORIGINAL_PROMPT_VD_RP = ROLE_VD + TASK_VD_RP + COT + INPUT

## ABLATION STUDY

# Variant 1. [TASK DESCRIPTION]. Think step by step, carefully. The input is [INPUT].
# Ablation component: ROLE_VD
PROMPT_VD_VARIANT_1 = TASK_VD + COT + INPUT

# Variant 2. You are [ROLE]. [TASK DESCRIPTION]. The input is [INPUT].
# Ablation component: COT
PROMPT_VD_VARIANT_2 = ROLE_VD + TASK_VD + INPUT

# Variant 3. [TASK DESCRIPTION]. The input is [INPUT]
# Ablation component: ROLE_VD, COT
PROMPT_VD_VARIANT_3 = TASK_VD + INPUT


## ADDITION STUDY

# Few-shot test 1
PROMPT_VD_FEW_SHOTS_1 = ROLE_VD + TASK_VD_FEW_SHOTS_1 + COT + INPUT
# Few-shot test 2
PROMPT_VD_FEW_SHOTS_2 = ROLE_VD + TASK_VD_FEW_SHOTS_2 + COT + INPUT
# Few-shot test 3
PROMPT_VD_FEW_SHOTS_3 = ROLE_VD + TASK_VD_FEW_SHOTS_3 + COT + INPUT


# Few-shot variant.
PROMPT_VD_FEW_SHOTS = ROLE_VD + TASK_VD_FEW_SHOTS + COT + INPUT
