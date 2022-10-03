import { Text, Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import { errorSeen, SystemState } from './systemSlice';

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    return {
      isConnected: system.isConnected,
      isProcessing: system.isProcessing,
      currentIteration: system.currentIteration,
      totalIterations: system.totalIterations,
      currentStatus: system.currentStatus,
      hasError: system.hasError,
      wasErrorSeen: system.wasErrorSeen,
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const StatusIndicator = () => {
  const {
    isConnected,
    isProcessing,
    currentIteration,
    totalIterations,
    currentStatus,
    hasError,
    wasErrorSeen,
  } = useAppSelector(systemSelector);
  const dispatch = useAppDispatch();
  // const statusMessageTextColor =
  //   isConnected && !hasError ? 'green.500' : 'red.500';

  let statusStyle;
  if (isConnected && !hasError) {
    statusStyle = 'status-good';
  } else {
    statusStyle = 'status-bad';
  }

  let statusMessage = currentStatus;

  const intermediateStatuses = [
    'generating',
    'preparing',
    'saving image',
    'restoring faces',
    'upscaling',
  ];

  if (intermediateStatuses.includes(statusMessage.toLowerCase())) {
    statusStyle = 'status-working';
  }

  if (statusMessage)
    if (isProcessing) {
      if (totalIterations > 1) {
        statusMessage += ` (${currentIteration}/${totalIterations})`;
      }
    }

  const tooltipLabel =
    hasError && !wasErrorSeen
      ? 'Click to clear, check logs for details'
      : undefined;

  const statusIndicatorCursor =
    hasError && !wasErrorSeen ? 'pointer' : 'initial';

  const handleClickStatusIndicator = () => {
    if (hasError || !wasErrorSeen) {
      dispatch(errorSeen());
    }
  };

  return (
    <Tooltip label={tooltipLabel}>
      <Text
        cursor={statusIndicatorCursor}
        onClick={handleClickStatusIndicator}
        className={`status ${statusStyle}`}
        // textColor={statusMessageTextColor}
      >
        {statusMessage}
      </Text>
    </Tooltip>
  );
};

export default StatusIndicator;
