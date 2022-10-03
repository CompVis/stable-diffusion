import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAISwitch from '../../../../common/components/IAISwitch';
import { setShouldFitToWidthHeight } from '../../optionsSlice';

export default function ImageFit() {
  const dispatch = useAppDispatch();

  const shouldFitToWidthHeight = useAppSelector(
    (state: RootState) => state.options.shouldFitToWidthHeight
  );

  const handleChangeFit = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldFitToWidthHeight(e.target.checked));

  return (
    <IAISwitch
      label="Fit initial image to output size"
      isChecked={shouldFitToWidthHeight}
      onChange={handleChangeFit}
    />
  );
}
