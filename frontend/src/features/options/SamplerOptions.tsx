import { Flex } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/store';

import {
  setCfgScale,
  setSampler,
  setSteps,
  OptionsState,
} from './optionsSlice';

import { SAMPLERS } from '../../app/constants';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';
import IAINumberInput from '../../common/components/IAINumberInput';
import IAISelect from '../../common/components/IAISelect';

const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      steps: options.steps,
      cfgScale: options.cfgScale,
      sampler: options.sampler,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Sampler options. Includes steps, CFG scale, sampler.
 */
const SamplerOptions = () => {
  const dispatch = useAppDispatch();
  const { steps, cfgScale, sampler } = useAppSelector(optionsSelector);

  const handleChangeSteps = (v: string | number) =>
    dispatch(setSteps(Number(v)));

  const handleChangeCfgScale = (v: string | number) =>
    dispatch(setCfgScale(Number(v)));

  const handleChangeSampler = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setSampler(e.target.value));

  return (
    <Flex gap={2} direction={'column'}>
      {/* <IAINumberInput
        label="Steps"
        min={1}
        step={1}
        precision={0}
        onChange={handleChangeSteps}
        value={steps}
      /> */}
      {/* <IAINumberInput
        label="CFG scale"
        step={0.5}
        onChange={handleChangeCfgScale}
        value={cfgScale}
      /> */}
      <IAISelect
        label="Sampler"
        value={sampler}
        onChange={handleChangeSampler}
        validValues={SAMPLERS}
      />
    </Flex>
  );
};

export default SamplerOptions;
