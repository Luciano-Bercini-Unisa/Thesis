/*
 * @source: https://smartcontractsecurity.github.io/SWC-registry/docs/SWC-118#incorrect-constructor-name2sol
 * @author: Ben Perez

 */

pragma solidity ^0.4.24;

contract Missing {
    address private owner;

    modifier onlyowner() {
        require(msg.sender == owner);
        _;
    }

    function Constructor() public {
        owner = msg.sender;
    }

    function() payable {}

    function withdraw() public onlyowner {
        owner.transfer(this.balance);
    }
}
