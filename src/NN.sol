// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {SSTORE2} from "lib/solmate/src/utils/SSTORE2.sol";

contract NN {
    uint256 constant MAX_STORAGE = 24_576 - 1; // 1 extra by for stop opcode~
    address[] public nnData;

    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function uploadNN(bytes calldata nn_) external {
        require(msg.sender == owner, "only owner");
        
        uint256 numContracts_ = nn_.length/MAX_STORAGE + 1;
        for (uint256 i_; i_ < numContracts_; i_++) {
            nnData.push(SSTORE2.write(
                nn_[i_ * MAX_STORAGE:nn_.length > (i_ + 1) * MAX_STORAGE ? (i_ + 1) * MAX_STORAGE : nn_.length]
            ));
        }
    }

    function getNNChunk(uint256 index_) external view returns(bytes memory){
        return SSTORE2.read(nnData[index_]);
    }
}
