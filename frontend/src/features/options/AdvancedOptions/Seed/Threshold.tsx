import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAINumberInput from '../../../../common/components/IAINumberInput';
import { setThreshold } from '../../optionsSlice';

export default function Threshold() {
  const dispatch = useAppDispatch();
  const threshold = useAppSelector(
    (state: RootState) => state.options.threshold
  );

  const handleChangeThreshold = (v: number) => dispatch(setThreshold(v));

  return (
    <IAINumberInput
      label="Threshold"
      min={0}
      max={1000}
      step={0.1}
      onChange={handleChangeThreshold}
      value={threshold}
      isInteger={false}
    />
  );
}
