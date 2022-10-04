import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAISwitch from '../../../../common/components/IAISwitch';
import { setShouldGenerateVariations } from '../../optionsSlice';

export default function GenerateVariations() {
  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.options.shouldGenerateVariations
  );

  const dispatch = useAppDispatch();

  const handleChangeShouldGenerateVariations = (
    e: ChangeEvent<HTMLInputElement>
  ) => dispatch(setShouldGenerateVariations(e.target.checked));

  return (
    <IAISwitch
      isChecked={shouldGenerateVariations}
      width={'auto'}
      onChange={handleChangeShouldGenerateVariations}
    />
  );
}
