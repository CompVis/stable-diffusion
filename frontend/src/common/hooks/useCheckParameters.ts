import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { useMemo } from 'react';
import { useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { OptionsState } from '../../features/options/optionsSlice';

import { SystemState } from '../../features/system/systemSlice';
import { validateSeedWeights } from '../util/seedWeightPairs';

export const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      prompt: options.prompt,
      shouldGenerateVariations: options.shouldGenerateVariations,
      seedWeights: options.seedWeights,
      maskPath: options.maskPath,
      initialImagePath: options.initialImagePath,
      seed: options.seed,
      activeTab: options.activeTab,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const systemSelector = createSelector(
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

/**
 * Checks relevant pieces of state to confirm generation will not deterministically fail.
 * This is used to prevent the 'Generate' button from being clicked.
 */
const useCheckParameters = (): boolean => {
  const { prompt } = useAppSelector(optionsSelector);

  const {
    shouldGenerateVariations,
    seedWeights,
    maskPath,
    initialImagePath,
    seed,
    activeTab,
  } = useAppSelector(optionsSelector);

  const { isProcessing, isConnected } = useAppSelector(systemSelector);

  return useMemo(() => {
    // Cannot generate without a prompt
    if (!prompt || Boolean(prompt.match(/^[\s\r\n]+$/))) {
      return false;
    }

    if (prompt && !initialImagePath && activeTab === 1) {
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
      (!(validateSeedWeights(seedWeights) || seedWeights === '') || seed === -1)
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
    activeTab,
  ]);
};

export default useCheckParameters;
