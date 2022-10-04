import React from 'react';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from '../../../../app/constants';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAINumberInput from '../../../../common/components/IAINumberInput';
import { setSeed } from '../../optionsSlice';

export default function Seed() {
  const seed = useAppSelector((state: RootState) => state.options.seed);
  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.options.shouldRandomizeSeed
  );
  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.options.shouldGenerateVariations
  );

  const dispatch = useAppDispatch();

  const handleChangeSeed = (v: number) => dispatch(setSeed(v));

  return (
    <IAINumberInput
      label="Seed"
      step={1}
      precision={0}
      flexGrow={1}
      min={NUMPY_RAND_MIN}
      max={NUMPY_RAND_MAX}
      isDisabled={shouldRandomizeSeed}
      isInvalid={seed < 0 && shouldGenerateVariations}
      onChange={handleChangeSeed}
      value={seed}
      width="10rem"
    />
  );
}
