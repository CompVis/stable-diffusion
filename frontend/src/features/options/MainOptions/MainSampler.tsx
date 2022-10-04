import React, { ChangeEvent } from 'react';
import { SAMPLERS } from '../../../app/constants';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAISelect from '../../../common/components/IAISelect';
import { setSampler } from '../optionsSlice';
import { fontSize } from './MainOptions';

export default function MainSampler() {
  const sampler = useAppSelector((state: RootState) => state.options.sampler);
  const dispatch = useAppDispatch();

  const handleChangeSampler = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setSampler(e.target.value));

  return (
    <IAISelect
      label="Sampler"
      value={sampler}
      onChange={handleChangeSampler}
      validValues={SAMPLERS}
      fontSize={fontSize}
      styleClass="main-option-block"
    />
  );
}
