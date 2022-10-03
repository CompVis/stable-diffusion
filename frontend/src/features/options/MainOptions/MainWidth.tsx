import React, { ChangeEvent } from 'react';
import { WIDTHS } from '../../../app/constants';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAISelect from '../../../common/components/IAISelect';
import { setWidth } from '../optionsSlice';
import { fontSize } from './MainOptions';

export default function MainWidth() {
  const width = useAppSelector((state: RootState) => state.options.width);
  const dispatch = useAppDispatch();

  const handleChangeWidth = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setWidth(Number(e.target.value)));

  return (
    <IAISelect
      label="Width"
      value={width}
      flexGrow={1}
      onChange={handleChangeWidth}
      validValues={WIDTHS}
      fontSize={fontSize}
      styleClass="main-option-block"
    />
  );
}
