import { Progress } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { SystemState } from '../system/systemSlice';

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      currentStep: system.currentStep,
      totalSteps: system.totalSteps,
      currentStatusHasSteps: system.currentStatusHasSteps,
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const ProgressBar = () => {
  const { isProcessing, currentStep, totalSteps, currentStatusHasSteps } =
    useAppSelector(systemSelector);

  const value = currentStep ? Math.round((currentStep * 100) / totalSteps) : 0;

  return (
    <Progress
      height="4px"
      value={value}
      isIndeterminate={isProcessing && !currentStatusHasSteps}
      className="progress-bar"
    />
  );
};

export default ProgressBar;
