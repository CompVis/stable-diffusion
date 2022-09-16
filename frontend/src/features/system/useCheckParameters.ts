import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { useMemo } from 'react';
import { useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { SDState } from '../sd/sdSlice';
import { validateSeedWeights } from '../sd/util/seedWeightPairs';
import { SystemState } from './systemSlice';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            prompt: sd.prompt,
            shouldGenerateVariations: sd.shouldGenerateVariations,
            seedWeights: sd.seedWeights,
            maskPath: sd.maskPath,
            initialImagePath: sd.initialImagePath,
            seed: sd.seed,
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
            isProcessing: system.isProcessing,
            isConnected: system.isConnected,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

/*
Checks relevant pieces of state to confirm generation will not deterministically fail.

This is used to prevent the 'Generate' button from being clicked.

Other parameter values may cause failure but we rely on input validation for those.
*/
const useCheckParameters = () => {
    const {
        prompt,
        shouldGenerateVariations,
        seedWeights,
        maskPath,
        initialImagePath,
        seed,
    } = useAppSelector(sdSelector);

    const { isProcessing, isConnected } = useAppSelector(systemSelector);

    return useMemo(() => {
        // Cannot generate without a prompt
        if (!prompt) {
            return false;
        }

        //  Cannot generate with a mask without img2img
        if (maskPath && !initialImagePath) {
            return false;
        }

        // TODO: job queue
        // Cannot generate if already processing an image
        if (isProcessing) {
            return false;
        }

        // Cannot generate if not connected
        if (!isConnected) {
            return false;
        }

        // Cannot generate variations without valid seed weights
        if (
            shouldGenerateVariations &&
            (!(validateSeedWeights(seedWeights) || seedWeights === '') ||
                seed === -1)
        ) {
            return false;
        }

        // All good
        return true;
    }, [
        prompt,
        maskPath,
        initialImagePath,
        isProcessing,
        isConnected,
        shouldGenerateVariations,
        seedWeights,
        seed,
    ]);
};

export default useCheckParameters;
