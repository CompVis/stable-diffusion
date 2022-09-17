import { Flex } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';

import { setHeight, setWidth, setSeamless, SDState } from '../sd/sdSlice';

import SDSelect from '../../components/SDSelect';

import { HEIGHTS, WIDTHS } from '../../app/constants';
import SDSwitch from '../../components/SDSwitch';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';

const sdSelector = createSelector(
  (state: RootState) => state.sd,
  (sd: SDState) => {
    return {
      height: sd.height,
      width: sd.width,
      seamless: sd.seamless,
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
  const { height, width, seamless } = useAppSelector(sdSelector);

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
