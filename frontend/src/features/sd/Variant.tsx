import {
    Flex,
    FormControl,
    FormLabel,
    HStack,
    Input,
    Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import SDNumberInput from '../../components/SDNumberInput';
import SDSwitch from '../../components/SDSwitch';
import {
    SDState,
    setSeedWeights,
    setShouldGenerateVariations,
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
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const Variant = () => {
    const { shouldGenerateVariations, variantAmount, seedWeights } =
        useAppSelector(sdSelector);

    const dispatch = useAppDispatch();

    return (
        <Flex gap={2} alignItems={'center'} pl={1}>
            <SDSwitch
                label='Generate variations'
                isChecked={shouldGenerateVariations}
                width={'auto'}
                onChange={(e) =>
                    dispatch(setShouldGenerateVariations(e.target.checked))
                }
            />
            <SDNumberInput
                label='Amount'
                value={variantAmount}
                step={0.01}
                min={0}
                max={1}
                width={240}
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
                        <Text fontSize={'sm'} whiteSpace='nowrap'>
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

export default Variant;
