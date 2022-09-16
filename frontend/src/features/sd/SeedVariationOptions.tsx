import {
    Flex,
    Input,
    HStack,
    FormControl,
    FormLabel,
    Text,
    Button,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from '../../app/constants';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import SDNumberInput from '../../components/SDNumberInput';
import SDSwitch from '../../components/SDSwitch';
import {
    randomizeSeed,
    SDState,
    setIterations,
    setSeed,
    setSeedWeights,
    setShouldGenerateVariations,
    setShouldRandomizeSeed,
    setVariantAmount,
} from './sdSlice';
import { validateSeedWeights } from './util/seedWeightPairs';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            variantAmount: sd.variantAmount,
            seedWeights: sd.seedWeights,
            shouldGenerateVariations: sd.shouldGenerateVariations,
            shouldRandomizeSeed: sd.shouldRandomizeSeed,
            seed: sd.seed,
            iterations: sd.iterations,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const SeedVariationOptions = () => {
    const {
        shouldGenerateVariations,
        variantAmount,
        seedWeights,
        shouldRandomizeSeed,
        seed,
        iterations,
    } = useAppSelector(sdSelector);

    const dispatch = useAppDispatch();

    return (
        <Flex gap={2} direction={'column'}>
            <SDNumberInput
                label='Images to generate'
                step={1}
                min={1}
                precision={0}
                onChange={(v) => dispatch(setIterations(Number(v)))}
                value={iterations}
            />
            <SDSwitch
                label='Randomize seed on generation'
                isChecked={shouldRandomizeSeed}
                onChange={(e) =>
                    dispatch(setShouldRandomizeSeed(e.target.checked))
                }
            />
            <Flex gap={2}>
                <SDNumberInput
                    label='Seed'
                    step={1}
                    precision={0}
                    flexGrow={1}
                    min={NUMPY_RAND_MIN}
                    max={NUMPY_RAND_MAX}
                    isDisabled={shouldRandomizeSeed}
                    isInvalid={seed < 0 && shouldGenerateVariations}
                    onChange={(v) => dispatch(setSeed(Number(v)))}
                    value={seed}
                />
                <Button
                    size={'sm'}
                    isDisabled={shouldRandomizeSeed}
                    onClick={() => dispatch(randomizeSeed())}
                >
                    <Text pl={2} pr={2}>
                        Shuffle
                    </Text>
                </Button>
            </Flex>
            <SDSwitch
                label='Generate variations'
                isChecked={shouldGenerateVariations}
                width={'auto'}
                onChange={(e) =>
                    dispatch(setShouldGenerateVariations(e.target.checked))
                }
            />
            <SDNumberInput
                label='Variation amount'
                value={variantAmount}
                step={0.01}
                min={0}
                max={1}
                isDisabled={!shouldGenerateVariations}
                onChange={(v) => dispatch(setVariantAmount(Number(v)))}
            />
            <FormControl
                isInvalid={
                    shouldGenerateVariations &&
                    !(validateSeedWeights(seedWeights) || seedWeights === '')
                }
                flexGrow={1}
                isDisabled={!shouldGenerateVariations}
            >
                <HStack>
                    <FormLabel marginInlineEnd={0} marginBottom={1}>
                        <Text whiteSpace='nowrap'>
                            Seed Weights
                        </Text>
                    </FormLabel>
                    <Input
                        size={'sm'}
                        value={seedWeights}
                        onChange={(e) =>
                            dispatch(setSeedWeights(e.target.value))
                        }
                    />
                </HStack>
            </FormControl>
        </Flex>
    );
};

export default SeedVariationOptions;
