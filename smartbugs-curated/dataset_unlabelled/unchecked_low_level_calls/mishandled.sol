/*
 * @source: https://github.com/seresistvanandras/EthBench/blob/master/Benchmark/Simple/mishandled.sol
 * @author: -

 */

pragma solidity ^0.4.0;
contract SendBack {
    mapping(address => uint) userBalances;
    function withdrawBalance() {
        uint amountToWithdraw = userBalances[msg.sender];
        userBalances[msg.sender] = 0;

        msg.sender.send(amountToWithdraw);
    }
}
