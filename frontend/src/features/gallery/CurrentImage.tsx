import { Center, Flex, Image, useColorModeValue } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { setAllParameters, setInitialImagePath, setSeed } from '../sd/sdSlice';
import { useState } from 'react';
import ImageMetadataViewer from './ImageMetadataViewer';
import DeleteImageModalButton from './DeleteImageModalButton';
import SDButton from '../../components/SDButton';
import { runESRGAN, runGFPGAN } from '../../app/socketio';
import { createSelector } from '@reduxjs/toolkit';
import { SystemState } from '../system/systemSlice';
import { isEqual } from 'lodash';

const height = 'calc(100vh - 238px)';

const systemSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => {
        return {
            isProcessing: system.isProcessing,
            isConnected: system.isConnected,
            isGFPGANAvailable: system.isGFPGANAvailable,
            isESRGANAvailable: system.isESRGANAvailable,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const CurrentImage = () => {
    const { currentImage, intermediateImage } = useAppSelector(
        (state: RootState) => state.gallery
    );
    const { isProcessing, isConnected, isGFPGANAvailable, isESRGANAvailable } =
        useAppSelector(systemSelector);

    const dispatch = useAppDispatch();

    const bgColor = useColorModeValue(
        'rgba(255, 255, 255, 0.85)',
        'rgba(0, 0, 0, 0.8)'
    );

    const [shouldShowImageDetails, setShouldShowImageDetails] =
        useState<boolean>(false);

    const imageToDisplay = intermediateImage || currentImage;

    return (
        <Flex direction={'column'} rounded={'md'} borderWidth={1} p={2} gap={2}>
            {imageToDisplay && (
                <Flex gap={2}>
                    <SDButton
                        label='Use as initial image'
                        colorScheme={'gray'}
                        flexGrow={1}
                        variant={'outline'}
                        onClick={() =>
                            dispatch(setInitialImagePath(imageToDisplay.url))
                        }
                    />

                    <SDButton
                        label='Use all'
                        colorScheme={'gray'}
                        flexGrow={1}
                        variant={'outline'}
                        onClick={() =>
                            dispatch(setAllParameters(imageToDisplay.metadata))
                        }
                    />

                    <SDButton
                        label='Use seed'
                        colorScheme={'gray'}
                        flexGrow={1}
                        variant={'outline'}
                        isDisabled={!imageToDisplay.metadata.seed}
                        onClick={() =>
                            dispatch(setSeed(imageToDisplay.metadata.seed!))
                        }
                    />

                    <SDButton
                        label='Upscale'
                        colorScheme={'gray'}
                        flexGrow={1}
                        variant={'outline'}
                        isDisabled={
                            !isESRGANAvailable ||
                            Boolean(intermediateImage) ||
                            !(isConnected && !isProcessing)
                        }
                        onClick={() => dispatch(runESRGAN(imageToDisplay))}
                    />
                    <SDButton
                        label='Fix faces'
                        colorScheme={'gray'}
                        flexGrow={1}
                        variant={'outline'}
                        isDisabled={
                            !isGFPGANAvailable ||
                            Boolean(intermediateImage) ||
                            !(isConnected && !isProcessing)
                        }
                        onClick={() => dispatch(runGFPGAN(imageToDisplay))}
                    />
                    <SDButton
                        label='Details'
                        colorScheme={'gray'}
                        variant={shouldShowImageDetails ? 'solid' : 'outline'}
                        borderWidth={1}
                        flexGrow={1}
                        onClick={() =>
                            setShouldShowImageDetails(!shouldShowImageDetails)
                        }
                    />
                    <DeleteImageModalButton image={imageToDisplay}>
                        <SDButton
                            label='Delete'
                            colorScheme={'red'}
                            flexGrow={1}
                            variant={'outline'}
                            isDisabled={Boolean(intermediateImage)}
                        />
                    </DeleteImageModalButton>
                </Flex>
            )}
            <Center height={height} position={'relative'}>
                {imageToDisplay && (
                    <Image
                        src={imageToDisplay.url}
                        fit='contain'
                        maxWidth={'100%'}
                        maxHeight={'100%'}
                    />
                )}
                {imageToDisplay && shouldShowImageDetails && (
                    <Flex
                        width={'100%'}
                        height={'100%'}
                        position={'absolute'}
                        top={0}
                        left={0}
                        p={3}
                        boxSizing='border-box'
                        backgroundColor={bgColor}
                        overflow='scroll'
                    >
                        <ImageMetadataViewer image={imageToDisplay} />
                    </Flex>
                )}
            </Center>
        </Flex>
    );
};

export default CurrentImage;
