import { Flex } from '@chakra-ui/react';

import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';

import { setHeight, setWidth, setSeamless, SDState } from '../sd/sdSlice';

import SDSelect from '../../components/SDSelect';

import { HEIGHTS, WIDTHS } from '../../app/constants';
import SDSwitch from '../../components/SDSwitch';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

const sdSelector = createSelector(
    (state: RootState) => state.sd,
    (sd: SDState) => {
        return {
            height: sd.height,
            width: sd.width,
            seamless: sd.seamless,
        };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const OutputOptions = () => {
    const { height, width, seamless } = useAppSelector(sdSelector);

    const dispatch = useAppDispatch();

    return (
        <Flex gap={2} direction={'column'}>
            <Flex gap={2}>
                <SDSelect
                    label='Width'
                    value={width}
                    flexGrow={1}
                    onChange={(e) => dispatch(setWidth(Number(e.target.value)))}
                    validValues={WIDTHS}
                />
                <SDSelect
                    label='Height'
                    value={height}
                    flexGrow={1}
                    onChange={(e) =>
                        dispatch(setHeight(Number(e.target.value)))
                    }
                    validValues={HEIGHTS}
                />
            </Flex>
            <SDSwitch
                label='Seamless tiling'
                fontSize={'md'}
                isChecked={seamless}
                onChange={(e) => dispatch(setSeamless(e.target.checked))}
            />
        </Flex>
    );
};

export default OutputOptions;
