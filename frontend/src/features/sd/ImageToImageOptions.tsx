import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import SDNumberInput from '../../components/SDNumberInput';
import SDSwitch from '../../components/SDSwitch';
import InitImage from './InitImage';
import {
    SDState,
    setImg2imgStrength,
    setShouldFitToWidthHeight,
} from './sdSlice';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            initialImagePath: sd.initialImagePath,
            img2imgStrength: sd.img2imgStrength,
            shouldFitToWidthHeight: sd.shouldFitToWidthHeight,
        };
    }
);

const ImageToImageOptions = () => {
    const { initialImagePath, img2imgStrength, shouldFitToWidthHeight } =
        useAppSelector(sdSelector);

    const dispatch = useAppDispatch();
    return (
        <Flex direction={'column'} gap={2}>
            <SDNumberInput
                isDisabled={!initialImagePath}
                label='Strength'
                step={0.01}
                min={0}
                max={1}
                onChange={(v) => dispatch(setImg2imgStrength(Number(v)))}
                value={img2imgStrength}
            />
            <SDSwitch
                isDisabled={!initialImagePath}
                label='Fit initial image to output size'
                isChecked={shouldFitToWidthHeight}
                onChange={(e) =>
                    dispatch(setShouldFitToWidthHeight(e.target.checked))
                }
            />
            <InitImage />
        </Flex>
    );
};

export default ImageToImageOptions;
