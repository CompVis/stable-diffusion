import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { cancelProcessing, generateImage } from '../../app/socketio/actions';
import { RootState } from '../../app/store';
import SDButton from '../../common/components/SDButton';
import useCheckParameters from '../../common/hooks/useCheckParameters';
import { SystemState } from '../system/systemSlice';

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

/**
 * Buttons to start and cancel image generation.
 */
const ProcessButtons = () => {
  const dispatch = useAppDispatch();
  const { isProcessing, isConnected } = useAppSelector(systemSelector);
  const isReady = useCheckParameters();

  const handleClickGenerate = () => dispatch(generateImage());

  const handleClickCancel = () => dispatch(cancelProcessing());

  return (
    <Flex
      gap={2}
      direction={'column'}
      alignItems={'space-between'}
      height={'100%'}
    >
      <SDButton
        label="Generate"
        type="submit"
        colorScheme="green"
        flexGrow={1}
        isDisabled={!isReady}
        fontSize={'md'}
        size={'md'}
        onClick={handleClickGenerate}
      />
      <SDButton
        label="Cancel"
        colorScheme="red"
        flexGrow={1}
        fontSize={'md'}
        size={'md'}
        isDisabled={!isConnected || !isProcessing}
        onClick={handleClickCancel}
      />
    </Flex>
  );
};

export default ProcessButtons;
