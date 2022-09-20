import { Flex } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/store';

import { setHeight, setWidth, setSeamless, OptionsState } from '../options/optionsSlice';


import { HEIGHTS, WIDTHS } from '../../app/constants';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';
import SDSelect from '../../common/components/SDSelect';
import SDSwitch from '../../common/components/SDSwitch';

const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      height: options.height,
      width: options.width,
      seamless: options.seamless,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Image output options. Includes width, height, seamless tiling.
 */
const OutputOptions = () => {
  const dispatch = useAppDispatch();
  const { height, width, seamless } = useAppSelector(optionsSelector);

  const handleChangeWidth = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setWidth(Number(e.target.value)));

  const handleChangeHeight = (e: ChangeEvent<HTMLSelectElement>) =>
    dispatch(setHeight(Number(e.target.value)));

  const handleChangeSeamless = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setSeamless(e.target.checked));

  return (
    <Flex gap={2} direction={'column'}>
      <Flex gap={2}>
        <SDSelect
          label="Width"
          value={width}
          flexGrow={1}
          onChange={handleChangeWidth}
          validValues={WIDTHS}
        />
        <SDSelect
          label="Height"
          value={height}
          flexGrow={1}
          onChange={handleChangeHeight}
          validValues={HEIGHTS}
        />
      </Flex>
      <SDSwitch
        label="Seamless tiling"
        fontSize={'md'}
        isChecked={seamless}
        onChange={handleChangeSeamless}
      />
    </Flex>
  );
};

export default OutputOptions;
