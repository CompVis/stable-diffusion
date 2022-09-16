import {
    Flex,
    Box,
    Text,
    Accordion,
    AccordionItem,
    AccordionButton,
    AccordionIcon,
    AccordionPanel,
    Switch,
} from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';

import {
    setShouldRunGFPGAN,
    setShouldRunESRGAN,
    SDState,
    setShouldUseInitImage,
} from '../sd/sdSlice';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { setOpenAccordions, SystemState } from '../system/systemSlice';
import SeedVariationOptions from './SeedVariationOptions';
import SamplerOptions from './SamplerOptions';
import ESRGANOptions from './ESRGANOptions';
import GFPGANOptions from './GFPGANOptions';
import OutputOptions from './OutputOptions';
import ImageToImageOptions from './ImageToImageOptions';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            initialImagePath: sd.initialImagePath,
            shouldUseInitImage: sd.shouldUseInitImage,
            shouldRunESRGAN: sd.shouldRunESRGAN,
            shouldRunGFPGAN: sd.shouldRunGFPGAN,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const systemSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => {
        return {
            isGFPGANAvailable: system.isGFPGANAvailable,
            isESRGANAvailable: system.isESRGANAvailable,
            openAccordions: system.openAccordions,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const OptionsAccordion = () => {
    const {
        shouldRunESRGAN,
        shouldRunGFPGAN,
        shouldUseInitImage,
        initialImagePath,
    } = useAppSelector(sdSelector);

    const { isGFPGANAvailable, isESRGANAvailable, openAccordions } =
        useAppSelector(systemSelector);

    const dispatch = useAppDispatch();

    return (
        <Accordion
            defaultIndex={openAccordions}
            allowMultiple
            reduceMotion
            onChange={(openAccordions) =>
                dispatch(setOpenAccordions(openAccordions))
            }
        >
            <AccordionItem>
                <h2>
                    <AccordionButton>
                        <Box flex='1' textAlign='left'>
                            Seed & Variation
                        </Box>
                        <AccordionIcon />
                    </AccordionButton>
                </h2>
                <AccordionPanel>
                    <SeedVariationOptions />
                </AccordionPanel>
            </AccordionItem>
            <AccordionItem>
                <h2>
                    <AccordionButton>
                        <Box flex='1' textAlign='left'>
                            Sampler
                        </Box>
                        <AccordionIcon />
                    </AccordionButton>
                </h2>
                <AccordionPanel>
                    <SamplerOptions />
                </AccordionPanel>
            </AccordionItem>
            <AccordionItem>
                <h2>
                    <AccordionButton>
                        <Flex
                            justifyContent={'space-between'}
                            alignItems={'center'}
                            width={'100%'}
                            mr={2}
                        >
                            <Text>Upscale (ESRGAN)</Text>
                            <Switch
                                isDisabled={!isESRGANAvailable}
                                isChecked={shouldRunESRGAN}
                                onChange={(e) =>
                                    dispatch(
                                        setShouldRunESRGAN(e.target.checked)
                                    )
                                }
                            />
                        </Flex>
                        <AccordionIcon />
                    </AccordionButton>
                </h2>
                <AccordionPanel>
                    <ESRGANOptions />
                </AccordionPanel>
            </AccordionItem>
            <AccordionItem>
                <h2>
                    <AccordionButton>
                        <Flex
                            justifyContent={'space-between'}
                            alignItems={'center'}
                            width={'100%'}
                            mr={2}
                        >
                            <Text>Fix Faces (GFPGAN)</Text>
                            <Switch
                                isDisabled={!isGFPGANAvailable}
                                isChecked={shouldRunGFPGAN}
                                onChange={(e) =>
                                    dispatch(
                                        setShouldRunGFPGAN(e.target.checked)
                                    )
                                }
                            />
                        </Flex>
                        <AccordionIcon />
                    </AccordionButton>
                </h2>
                <AccordionPanel>
                    <GFPGANOptions />
                </AccordionPanel>
            </AccordionItem>
            <AccordionItem>
                <h2>
                    <AccordionButton>
                        <Flex
                            justifyContent={'space-between'}
                            alignItems={'center'}
                            width={'100%'}
                            mr={2}
                        >
                            <Text>Image to Image</Text>
                            <Switch
                                isDisabled={!initialImagePath}
                                isChecked={shouldUseInitImage}
                                onChange={(e) =>
                                    dispatch(
                                        setShouldUseInitImage(e.target.checked)
                                    )
                                }
                            />
                        </Flex>
                        <AccordionIcon />
                    </AccordionButton>
                </h2>
                <AccordionPanel>
                    <ImageToImageOptions />
                </AccordionPanel>
            </AccordionItem>
            <AccordionItem>
                <h2>
                    <AccordionButton>
                        <Box flex='1' textAlign='left'>
                            Output
                        </Box>
                        <AccordionIcon />
                    </AccordionButton>
                </h2>
                <AccordionPanel>
                    <OutputOptions />
                </AccordionPanel>
            </AccordionItem>
        </Accordion>
    );
};

export default OptionsAccordion;
