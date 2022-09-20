import * as InvokeAI from '../../app/invokeai';

export const stringToSeedWeights = (
  string: string
): InvokeAI.SeedWeights | boolean => {
  const stringPairs = string.split(',');
  const arrPairs = stringPairs.map((p) => p.split(':'));
  const pairs = arrPairs.map((p: Array<string>): InvokeAI.SeedWeightPair => {
    return { seed: parseInt(p[0]), weight: parseFloat(p[1]) };
  });

  if (!validateSeedWeights(pairs)) {
    return false;
  }

  return pairs;
};

export const validateSeedWeights = (
  seedWeights: InvokeAI.SeedWeights | string
): boolean => {
  return typeof seedWeights === 'string'
    ? Boolean(stringToSeedWeights(seedWeights))
    : Boolean(
        seedWeights.length &&
          !seedWeights.some((pair: InvokeAI.SeedWeightPair) => {
            const { seed, weight } = pair;
            const isSeedValid = !isNaN(parseInt(seed.toString(), 10));
            const isWeightValid =
              !isNaN(parseInt(weight.toString(), 10)) &&
              weight >= 0 &&
              weight <= 1;
            return !(isSeedValid && isWeightValid);
          })
      );
};

export const seedWeightsToString = (
  seedWeights: InvokeAI.SeedWeights
): string => {
  return seedWeights.reduce((acc, pair, i, arr) => {
    const { seed, weight } = pair;
    acc += `${seed}:${weight}`;
    if (i !== arr.length - 1) {
      acc += ',';
    }
    return acc;
  }, '');
};

export const seedWeightsToArray = (
  seedWeights: InvokeAI.SeedWeights
): Array<Array<number>> => {
  return seedWeights.map((pair: InvokeAI.SeedWeightPair) => [
    pair.seed,
    pair.weight,
  ]);
};

export const stringToSeedWeightsArray = (
  string: string
): Array<Array<number>> => {
  const stringPairs = string.split(',');
  const arrPairs = stringPairs.map((p) => p.split(':'));
  return arrPairs.map(
    (p: Array<string>): Array<number> => [parseInt(p[0]), parseFloat(p[1])]
  );
};
