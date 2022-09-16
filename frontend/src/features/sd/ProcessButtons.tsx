import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { cancelProcessing, generateImage } from '../../app/socketio';
import { RootState } from '../../app/store';
import SDButton from '../../components/SDButton';
import { SystemState } from '../system/systemSlice';
import useCheckParameters from '../system/useCheckParameters';

const systemSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => {
        return {
            isProcessing: system.isProcessing,
            isConnected: system.isConnected,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const ProcessButtons = () => {
    const { isProcessing, isConnected } = useAppSelector(systemSelector);

    const dispatch = useAppDispatch();

    const isReady = useCheckParameters();

    return (
        <Flex gap={2} direction={'column'} alignItems={'space-between'} height={'100%'}>
            <SDButton
                label='Generate'
                type='submit'
                colorScheme='green'
                flexGrow={1}
                isDisabled={!isReady}
                fontSize={'md'}
                size={'md'}
                onClick={() => dispatch(generateImage())}
            />
            <SDButton
                label='Cancel'
                colorScheme='red'
                flexGrow={1}
                fontSize={'md'}
                size={'md'}
                isDisabled={!isConnected || !isProcessing}
                onClick={() => dispatch(cancelProcessing())}
            />
        </Flex>
    );
};

export default ProcessButtons;
