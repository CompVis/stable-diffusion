import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAINumberInput from '../../../../common/components/IAINumberInput';
import { setImg2imgStrength } from '../../optionsSlice';

interface ImageToImageStrengthProps {
  label?: string;
  styleClass?: string;
}

export default function ImageToImageStrength(props: ImageToImageStrengthProps) {
  const { label = 'Strength', styleClass } = props;
  const img2imgStrength = useAppSelector(
    (state: RootState) => state.options.img2imgStrength
  );

  const dispatch = useAppDispatch();

  const handleChangeStrength = (v: number) => dispatch(setImg2imgStrength(v));

  return (
    <IAINumberInput
      label={label}
      step={0.01}
      min={0.01}
      max={0.99}
      onChange={handleChangeStrength}
      value={img2imgStrength}
      width="90px"
      isInteger={false}
      styleClass={styleClass}
    />
  );
}
