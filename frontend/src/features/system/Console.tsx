import { IconButton, Tooltip } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { errorSeen, setShouldShowLogViewer, SystemState } from './systemSlice';
import { useLayoutEffect, useRef, useState } from 'react';
import { FaAngleDoubleDown, FaCode, FaMinus } from 'react-icons/fa';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { Resizable } from 're-resizable';
import { useHotkeys } from 'react-hotkeys-hook';

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
    return {
      shouldShowLogViewer: system.shouldShowLogViewer,
      hasError: system.hasError,
      wasErrorSeen: system.wasErrorSeen,
    };
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
const Console = () => {
  const dispatch = useAppDispatch();
  const log = useAppSelector(logSelector);
  const { shouldShowLogViewer, hasError, wasErrorSeen } =
    useAppSelector(systemSelector);

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
    dispatch(errorSeen());
    dispatch(setShouldShowLogViewer(!shouldShowLogViewer));
  };

  useHotkeys(
    '`',
    () => {
      dispatch(setShouldShowLogViewer(!shouldShowLogViewer));
    },
    [shouldShowLogViewer]
  );

  return (
    <>
      {shouldShowLogViewer && (
        <Resizable
          defaultSize={{
            width: '100%',
            height: 200,
          }}
          style={{ display: 'flex', position: 'fixed', left: 0, bottom: 0 }}
          maxHeight={'90vh'}
        >
          <div className="console" ref={viewerRef}>
            {log.map((entry, i) => {
              const { timestamp, message, level } = entry;
              return (
                <div key={i} className={`console-entry console-${level}-color`}>
                  <p className="console-timestamp">{timestamp}:</p>
                  <p className="console-message">{message}</p>
                </div>
              );
            })}
          </div>
        </Resizable>
      )}
      {shouldShowLogViewer && (
        <Tooltip hasArrow label={shouldAutoscroll ? 'Autoscroll On' : 'Autoscroll Off'}>
          <IconButton
            className={`console-autoscroll-icon-button ${
              shouldAutoscroll && 'autoscroll-enabled'
            }`}
            size="sm"
            aria-label="Toggle autoscroll"
            variant={'solid'}
            icon={<FaAngleDoubleDown />}
            onClick={() => setShouldAutoscroll(!shouldAutoscroll)}
          />
        </Tooltip>
      )}
      <Tooltip hasArrow label={shouldShowLogViewer ? 'Hide Console' : 'Show Console'}>
        <IconButton
          className={`console-toggle-icon-button ${
            (hasError || !wasErrorSeen) && 'error-seen'
          }`}
          size="sm"
          position={'fixed'}
          variant={'solid'}
          aria-label="Toggle Log Viewer"
          // colorScheme={hasError || !wasErrorSeen ? 'red' : 'gray'}
          icon={shouldShowLogViewer ? <FaMinus /> : <FaCode />}
          onClick={handleClickLogViewerToggle}
        />
      </Tooltip>
    </>
  );
};

export default Console;
