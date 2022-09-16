import { Flex } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';

import { SDState, setGfpganStrength } from '../sd/sdSlice';

import SDNumberInput from '../../components/SDNumberInput';

import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { SystemState } from '../system/systemSlice';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            gfpganStrength: sd.gfpganStrength,
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
            isGFPGANAvailable: system.isGFPGANAvailable,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);
const GFPGANOptions = () => {
    const { gfpganStrength } = useAppSelector(sdSelector);

    const { isGFPGANAvailable } = useAppSelector(systemSelector);

    const dispatch = useAppDispatch();

    return (
        <Flex direction={'column'} gap={2}>
            <SDNumberInput
                isDisabled={!isGFPGANAvailable}
                label='Strength'
                step={0.05}
                min={0}
                max={1}
                onChange={(v) => dispatch(setGfpganStrength(Number(v)))}
                value={gfpganStrength}
            />
        </Flex>
    );
};

export default GFPGANOptions;
