import { Flex } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';

import {
    setUpscalingLevel,
    setUpscalingStrength,
    UpscalingLevel,
    SDState,
} from '../sd/sdSlice';

import SDNumberInput from '../../components/SDNumberInput';
import SDSelect from '../../components/SDSelect';

import { UPSCALING_LEVELS } from '../../app/constants';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { SystemState } from '../system/systemSlice';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            upscalingLevel: sd.upscalingLevel,
            upscalingStrength: sd.upscalingStrength,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const systemSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => {
        return {
            isESRGANAvailable: system.isESRGANAvailable,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);
const ESRGANOptions = () => {
    const { upscalingLevel, upscalingStrength } = useAppSelector(sdSelector);

    const { isESRGANAvailable } = useAppSelector(systemSelector);

    const dispatch = useAppDispatch();

    return (
        <Flex direction={'column'} gap={2}>
            <SDSelect
                isDisabled={!isESRGANAvailable}
                label='Scale'
                value={upscalingLevel}
                onChange={(e) =>
                    dispatch(
                        setUpscalingLevel(
                            Number(e.target.value) as UpscalingLevel
                        )
                    )
                }
                validValues={UPSCALING_LEVELS}
            />
            <SDNumberInput
                isDisabled={!isESRGANAvailable}
                label='Strength'
                step={0.05}
                min={0}
                max={1}
                onChange={(v) => dispatch(setUpscalingStrength(Number(v)))}
                value={upscalingStrength}
            />
        </Flex>
    );
};

export default ESRGANOptions;
