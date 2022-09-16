import { Progress } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { SDState } from '../sd/sdSlice';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            realSteps: sd.realSteps,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const ProgressBar = () => {
    const { realSteps } = useAppSelector(sdSelector);
    const { currentStep } = useAppSelector((state: RootState) => state.system);
    const progress = Math.round((currentStep * 100) / realSteps);
    return (
        <Progress
            height='10px'
            value={progress}
            isIndeterminate={progress < 0 || currentStep === realSteps}
        />
    );
};

export default ProgressBar;
