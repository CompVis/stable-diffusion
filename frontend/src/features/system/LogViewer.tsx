import {
    IconButton,
    useColorModeValue,
    Flex,
    Text,
    Tooltip,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { setShouldShowLogViewer, SystemState } from './systemSlice';
import { useLayoutEffect, useRef, useState } from 'react';
import { FaAngleDoubleDown, FaCode, FaMinus } from 'react-icons/fa';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

const logSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => system.log,
    {
        memoizeOptions: {
            resultEqualityCheck: (a, b) => a.length === b.length,
        },
    }
);

const systemSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => {
        return { shouldShowLogViewer: system.shouldShowLogViewer };
    },
    {
        memoizeOptions: {
            resultEqualityCheck: isEqual,
        },
    }
);

const LogViewer = () => {
    const dispatch = useAppDispatch();
    const bg = useColorModeValue('gray.50', 'gray.900');
    const borderColor = useColorModeValue('gray.500', 'gray.500');
    const [shouldAutoscroll, setShouldAutoscroll] = useState<boolean>(true);

    const log = useAppSelector(logSelector);
    const { shouldShowLogViewer } = useAppSelector(systemSelector);

    const viewerRef = useRef<HTMLDivElement>(null);

    useLayoutEffect(() => {
        if (viewerRef.current !== null && shouldAutoscroll) {
            viewerRef.current.scrollTop = viewerRef.current.scrollHeight;
        }
    });

    return (
        <>
            {shouldShowLogViewer && (
                <Flex
                    position={'fixed'}
                    left={0}
                    bottom={0}
                    height='200px'
                    width='100vw'
                    overflow='auto'
                    direction='column'
                    fontFamily='monospace'
                    fontSize='sm'
                    pl={12}
                    pr={2}
                    pb={2}
                    borderTopWidth='4px'
                    borderColor={borderColor}
                    background={bg}
                    ref={viewerRef}
                >
                    {log.map((entry, i) => (
                        <Flex gap={2} key={i}>
                            <Text fontSize='sm' fontWeight={'semibold'}>
                                {entry.timestamp}:
                            </Text>
                            <Text fontSize='sm' wordBreak={'break-all'}>
                                {entry.message}
                            </Text>
                        </Flex>
                    ))}
                </Flex>
            )}
            {shouldShowLogViewer && (
                <Tooltip
                    label={
                        shouldAutoscroll ? 'Autoscroll on' : 'Autoscroll off'
                    }
                >
                    <IconButton
                        size='sm'
                        position={'fixed'}
                        left={2}
                        bottom={12}
                        aria-label='Toggle autoscroll'
                        variant={'solid'}
                        colorScheme={shouldAutoscroll ? 'blue' : 'gray'}
                        icon={<FaAngleDoubleDown />}
                        onClick={() => setShouldAutoscroll(!shouldAutoscroll)}
                    />
                </Tooltip>
            )}
            <Tooltip label={shouldShowLogViewer ? 'Hide logs' : 'Show logs'}>
                <IconButton
                    size='sm'
                    position={'fixed'}
                    left={2}
                    bottom={2}
                    variant={'solid'}
                    aria-label='Toggle Log Viewer'
                    icon={shouldShowLogViewer ? <FaMinus /> : <FaCode />}
                    onClick={() =>
                        dispatch(setShouldShowLogViewer(!shouldShowLogViewer))
                    }
                />
            </Tooltip>
        </>
    );
};

export default LogViewer;
