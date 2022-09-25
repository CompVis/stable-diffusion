import { Flex, Spinner } from '@chakra-ui/react';

const Loading = () => {
    return (
        <Flex
            width={'100vw'}
            height={'100vh'}
            alignItems='center'
            justifyContent='center'
        >
            <Spinner
                thickness='2px'
                speed='1s'
                emptyColor='gray.200'
                color='gray.400'
                size='xl'
            />
        </Flex>
    );
};

export default Loading;
