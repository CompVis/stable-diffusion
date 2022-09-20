import { Flex } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/store';

import {
  setUpscalingLevel,
  setUpscalingStrength,
  UpscalingLevel,
  OptionsState,
} from '../options/optionsSlice';


import { UPSCALING_LEVELS } from '../../app/constants';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { SystemState } from '../system/systemSlice';
import { ChangeEvent } from 'react';
import SDNumberInput from '../../common/components/SDNumberInput';
import SDSelect from '../../common/components/SDSelect';

const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      upscalingLevel: options.upscalingLevel,
      upscalingStrength: options.upscalingStrength,
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

/**
 * Displays upscaling/ESRGAN options (level and strength).
 */
const ESRGANOptions = () => {
  const dispatch = useAppDispatch();
  const { upscalingLevel, upscalingStrength } = useAppSelector(optionsSelector);
  const { isESRGANAvailable } = useAppSelector(systemSelector);

  const handleChangeLevel = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setUpscalingLevel(Number(e.target.value) as UpscalingLevel));

  const handleChangeStrength = (v: string | number) =>
    dispatch(setUpscalingStrength(Number(v)));

  return (
    <Flex direction={'column'} gap={2}>
      <SDSelect
        isDisabled={!isESRGANAvailable}
        label="Scale"
        value={upscalingLevel}
        onChange={handleChangeLevel}
        validValues={UPSCALING_LEVELS}
      />
      <SDNumberInput
        isDisabled={!isESRGANAvailable}
        label="Strength"
        step={0.05}
        min={0}
        max={1}
        onChange={handleChangeStrength}
        value={upscalingStrength}
      />
    </Flex>
  );
};

export default ESRGANOptions;
