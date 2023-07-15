// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "forge-std/Script.sol";

import "forge-std/Test.sol";

import "../src/NN.sol";

contract Deploy is Script, Test {
    function setUp() public {}

    function run() public {
        vm.startBroadcast();
        NN nn_ = new NN();
        console.log("nn address: ", address(nn_));
        for (uint256 i_; i_ < 1; i_++) {
            nn_.uploadNN(bytes(vm.readFile(string(abi.encodePacked("model_chunks/model_data_", vm.toString(i_), ".json")))));
        }        
        vm.stopBroadcast();
        _verify(address(nn_));
    }

    function _verify(address nn_) internal {
        for (uint256 i_; i_ < 1; i_++) {
            string memory chunk_ = string(abi.encodePacked("model_chunks/model_data_", vm.toString(i_), ".json"));
            assertEq(vm.readFile(chunk_),  NN(nn_).getNNChunk(i_));
        }
    }
}
