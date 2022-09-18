import {
  IconButton,
  useColorModeValue,
  Flex,
  Text,
  Tooltip,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from '../../app/store';
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
      // We don't need a deep equality check for this selector.
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

/**
 * Basic log viewer, floats on bottom of page.
 */
const LogViewer = () => {
  const dispatch = useAppDispatch();
  const log = useAppSelector(logSelector);
  const { shouldShowLogViewer } = useAppSelector(systemSelector);

  // Set colors based on dark/light mode
  const bg = useColorModeValue('gray.50', 'gray.900');
  const borderColor = useColorModeValue('gray.500', 'gray.500');
  const logTextColors = useColorModeValue(
    {
      info: undefined,
      warning: 'yellow.500',
      error: 'red.500',
    },
    {
      info: undefined,
      warning: 'yellow.300',
      error: 'red.300',
    }
  );

  // Rudimentary autoscroll
  const [shouldAutoscroll, setShouldAutoscroll] = useState<boolean>(true);
  const viewerRef = useRef<HTMLDivElement>(null);

  /**
   * If autoscroll is on, scroll to the bottom when:
   * - log updates
   * - viewer is toggled
   *
   * Also scroll to the bottom whenever autoscroll is turned on.
   */
  useLayoutEffect(() => {
    if (viewerRef.current !== null && shouldAutoscroll) {
      viewerRef.current.scrollTop = viewerRef.current.scrollHeight;
    }
  }, [shouldAutoscroll, log, shouldShowLogViewer]);

  const handleClickLogViewerToggle = () => {
    dispatch(setShouldShowLogViewer(!shouldShowLogViewer));
  };

  return (
    <>
      {shouldShowLogViewer && (
        <Flex
          position={'fixed'}
          left={0}
          bottom={0}
          height="200px" // TODO: Make the log viewer resizeable.
          width="100vw"
          overflow="auto"
          direction="column"
          fontFamily="monospace"
          fontSize="sm"
          pl={12}
          pr={2}
          pb={2}
          borderTopWidth="4px"
          borderColor={borderColor}
          background={bg}
          ref={viewerRef}
        >
          {log.map((entry, i) => {
            const { timestamp, message, level } = entry;
            return (
              <Flex gap={2} key={i} textColor={logTextColors[level]}>
                <Text fontSize="sm" fontWeight={'semibold'}>
                  {timestamp}:
                </Text>
                <Text fontSize="sm" wordBreak={'break-all'}>
                  {message}
                </Text>
              </Flex>
            );
          })}
        </Flex>
      )}
      {shouldShowLogViewer && (
        <Tooltip label={shouldAutoscroll ? 'Autoscroll on' : 'Autoscroll off'}>
          <IconButton
            size="sm"
            position={'fixed'}
            left={2}
            bottom={12}
            aria-label="Toggle autoscroll"
            variant={'solid'}
            colorScheme={shouldAutoscroll ? 'blue' : 'gray'}
            icon={<FaAngleDoubleDown />}
            onClick={() => setShouldAutoscroll(!shouldAutoscroll)}
          />
        </Tooltip>
      )}
      <Tooltip label={shouldShowLogViewer ? 'Hide logs' : 'Show logs'}>
        <IconButton
          size="sm"
          position={'fixed'}
          left={2}
          bottom={2}
          variant={'solid'}
          aria-label="Toggle Log Viewer"
          icon={shouldShowLogViewer ? <FaMinus /> : <FaCode />}
          onClick={handleClickLogViewerToggle}
        />
      </Tooltip>
    </>
  );
};

export default LogViewer;
