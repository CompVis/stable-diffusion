import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAINumberInput from '../../../../common/components/IAINumberInput';
import { setPerlin } from '../../optionsSlice';

export default function Perlin() {
  const dispatch = useAppDispatch();
  const perlin = useAppSelector((state: RootState) => state.options.perlin);

  const handleChangePerlin = (v: number) => dispatch(setPerlin(v));

  return (
    <IAINumberInput
      label="Perlin Noise"
      min={0}
      max={1}
      step={0.05}
      onChange={handleChangePerlin}
      value={perlin}
      isInteger={false}
    />
  );
}
