export interface SeedWeightPair {
    seed: number;
    weight: number;
}

export type SeedWeights = Array<Array<number>>;

export const stringToSeedWeights = (string: string): SeedWeights | boolean => {
    const stringPairs = string.split(',');
    const arrPairs = stringPairs.map((p) => p.split(':'));
    const pairs = arrPairs.map((p) => {
        return [parseInt(p[0]), parseFloat(p[1])];
    });

    if (!validateSeedWeights(pairs)) {
        return false;
    }

    return pairs;
};

export const validateSeedWeights = (
    seedWeights: SeedWeights | string
): boolean => {
    return typeof seedWeights === 'string'
        ? Boolean(stringToSeedWeights(seedWeights))
        : Boolean(
              seedWeights.length &&
                  !seedWeights.some((pair) => {
                      const [seed, weight] = pair;
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
    seedWeights: SeedWeights
): string | boolean => {
    if (!validateSeedWeights(seedWeights)) {
        return false;
    }

    return seedWeights.reduce((acc, pair, i, arr) => {
        const [seed, weight] = pair;
        acc += `${seed}:${weight}`;
        if (i !== arr.length - 1) {
            acc += ',';
        }
        return acc;
    }, '');
};
