import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { ChangeEvent } from 'react';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import SDNumberInput from '../../common/components/SDNumberInput';
import SDSwitch from '../../common/components/SDSwitch';
import InitAndMaskImage from './InitAndMaskImage';
import {
  OptionsState,
  setImg2imgStrength,
  setShouldFitToWidthHeight,
} from './optionsSlice';

const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      img2imgStrength: options.img2imgStrength,
      shouldFitToWidthHeight: options.shouldFitToWidthHeight,
    };
  }
);

/**
 * Options for img2img generation (strength, fit, init/mask upload).
 */
const ImageToImageOptions = () => {
  const dispatch = useAppDispatch();
  const { img2imgStrength, shouldFitToWidthHeight } =
    useAppSelector(optionsSelector);

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
