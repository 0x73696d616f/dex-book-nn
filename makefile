-include .env

deploy-apothem :;
	forge script script/DeployNN.s.sol:Deploy --broadcast --rpc-url ${RPC_URL_APOTHEM} --private-key ${PRIVATE_KEY} --etherscan-api-key abc --verifier-url https://explorer.apothem.network/api --verify --delay 20 --retries 10 --legacy -vv 
