import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { ChangeEvent } from 'react';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import SDNumberInput from '../../common/components/SDNumberInput';
import SDSwitch from '../../common/components/SDSwitch';
import InitAndMaskImage from './InitAndMaskImage';
import {
  SDState,
  setImg2imgStrength,
  setShouldFitToWidthHeight,
} from './sdSlice';

const sdSelector = createSelector(
  (state: RootState) => state.sd,
  (sd: SDState) => {
    return {
      img2imgStrength: sd.img2imgStrength,
      shouldFitToWidthHeight: sd.shouldFitToWidthHeight,
    };
  }
);

/**
 * Options for img2img generation (strength, fit, init/mask upload).
 */
const ImageToImageOptions = () => {
  const dispatch = useAppDispatch();
  const { img2imgStrength, shouldFitToWidthHeight } =
    useAppSelector(sdSelector);

  const handleChangeStrength = (v: string | number) =>
    dispatch(setImg2imgStrength(Number(v)));

  const handleChangeFit = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldFitToWidthHeight(e.target.checked));

  return (
    <Flex direction={'column'} gap={2}>
      <SDNumberInput
        label="Strength"
        step={0.01}
        min={0}
        max={1}
        onChange={handleChangeStrength}
        value={img2imgStrength}
      />
      <SDSwitch
        label="Fit initial image to output size"
        isChecked={shouldFitToWidthHeight}
        onChange={handleChangeFit}
      />
      <InitAndMaskImage />
    </Flex>
  );
};

export default ImageToImageOptions;
