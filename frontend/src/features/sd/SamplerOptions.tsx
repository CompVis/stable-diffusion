import { Flex } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/store';

import { setCfgScale, setSampler, setSteps, SDState } from '../sd/sdSlice';


import { SAMPLERS } from '../../app/constants';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';
import SDNumberInput from '../../common/components/SDNumberInput';
import SDSelect from '../../common/components/SDSelect';

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

/**
 * Sampler options. Includes steps, CFG scale, sampler.
 */
const SamplerOptions = () => {
  const dispatch = useAppDispatch();
  const { steps, cfgScale, sampler } = useAppSelector(sdSelector);

  const handleChangeSteps = (v: string | number) =>
    dispatch(setSteps(Number(v)));

  const handleChangeCfgScale = (v: string | number) =>
    dispatch(setCfgScale(Number(v)));

  const handleChangeSampler = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setSampler(e.target.value));

  return (
    <Flex gap={2} direction={'column'}>
      <SDNumberInput
        label="Steps"
        min={1}
        step={1}
        precision={0}
        onChange={handleChangeSteps}
        value={steps}
      />
      <SDNumberInput
        label="CFG scale"
        step={0.5}
        onChange={handleChangeCfgScale}
        value={cfgScale}
      />
      <SDSelect
        label="Sampler"
        value={sampler}
        onChange={handleChangeSampler}
        validValues={SAMPLERS}
      />
    </Flex>
  );
};

export default SamplerOptions;
