import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAINumberInput from '../../../../common/components/IAINumberInput';
import { setVariationAmount } from '../../optionsSlice';

export default function VariationAmount() {
  const variationAmount = useAppSelector(
    (state: RootState) => state.options.variationAmount
  );

  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.options.shouldGenerateVariations
  );

  const dispatch = useAppDispatch();
  const handleChangevariationAmount = (v: number) =>
    dispatch(setVariationAmount(v));

  return (
    <IAINumberInput
      label="Variation Amount"
      value={variationAmount}
      step={0.01}
      min={0}
      max={1}
      isDisabled={!shouldGenerateVariations}
      onChange={handleChangevariationAmount}
      isInteger={false}
    />
  );
}
