import { Flex } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';

import { setCfgScale, setSampler, setSteps, SDState } from '../sd/sdSlice';

import SDNumberInput from '../../components/SDNumberInput';
import SDSelect from '../../components/SDSelect';

import { SAMPLERS } from '../../app/constants';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            steps: sd.steps,
            cfgScale: sd.cfgScale,
            sampler: sd.sampler,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const SamplerOptions = () => {
    const { steps, cfgScale, sampler } = useAppSelector(sdSelector);

    const dispatch = useAppDispatch();

    return (
        <Flex gap={2} direction={'column'}>
            <SDNumberInput
                label='Steps'
                min={1}
                step={1}
                precision={0}
                onChange={(v) => dispatch(setSteps(Number(v)))}
                value={steps}
            />
            <SDNumberInput
                label='CFG scale'
                step={0.5}
                onChange={(v) => dispatch(setCfgScale(Number(v)))}
                value={cfgScale}
            />
            <SDSelect
                label='Sampler'
                value={sampler}
                onChange={(e) => dispatch(setSampler(e.target.value))}
                validValues={SAMPLERS}
            />
        </Flex>
    );
};

export default SamplerOptions;
