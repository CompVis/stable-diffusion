import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { ExpandedIndex } from '@chakra-ui/react';

export type LogLevel = 'info' | 'warning' | 'error';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
}

export interface Log {
  [index: number]: LogEntry;
}

export interface SystemStatus {
  isProcessing: boolean;
  currentStep: number;
  totalSteps: number;
  currentIteration: number;
  totalIterations: number;
  currentStatus: string;
  currentStatusHasSteps: boolean;
}

export interface SystemState extends SystemStatus {
  shouldDisplayInProgress: boolean;
  log: Array<LogEntry>;
  shouldShowLogViewer: boolean;
  isGFPGANAvailable: boolean;
  isESRGANAvailable: boolean;
  isConnected: boolean;
  socketId: string;
  shouldConfirmOnDelete: boolean;
  openAccordions: ExpandedIndex;
  currentStep: number;
  totalSteps: number;
  currentIteration: number;
  totalIterations: number;
  currentStatus: string;
  currentStatusHasSteps: boolean;
}

const initialSystemState = {
  isConnected: false,
  isProcessing: false,
  log: [],
  shouldShowLogViewer: false,
  shouldDisplayInProgress: false,
  isGFPGANAvailable: true,
  isESRGANAvailable: true,
  socketId: '',
  shouldConfirmOnDelete: true,
  openAccordions: [0],
  currentStep: 0,
  totalSteps: 0,
  currentIteration: 0,
  totalIterations: 0,
  currentStatus: '',
  currentStatusHasSteps: false,
};

const initialState: SystemState = initialSystemState;

export const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setShouldDisplayInProgress: (state, action: PayloadAction<boolean>) => {
      state.shouldDisplayInProgress = action.payload;
    },
    setIsProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
    },
    setCurrentStatus: (state, action: PayloadAction<string>) => {
      state.currentStatus = action.payload;
    },
    setSystemStatus: (state, action: PayloadAction<SystemStatus>) => {
      const currentStatus =
        !action.payload.isProcessing && state.isConnected
          ? 'Connected'
          : action.payload.currentStatus;

      return { ...state, ...action.payload, currentStatus };
    },
    addLogEntry: (
      state,
      action: PayloadAction<{
        timestamp: string;
        message: string;
        level?: LogLevel;
      }>
    ) => {
      const { timestamp, message, level } = action.payload;
      const logLevel = level || 'info';

      const entry: LogEntry = {
        timestamp,
        message,
        level: logLevel,
      };

      state.log.push(entry);
    },
    setShouldShowLogViewer: (state, action: PayloadAction<boolean>) => {
      state.shouldShowLogViewer = action.payload;
    },
    setIsConnected: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
    setSocketId: (state, action: PayloadAction<string>) => {
      state.socketId = action.payload;
    },
    setShouldConfirmOnDelete: (state, action: PayloadAction<boolean>) => {
      state.shouldConfirmOnDelete = action.payload;
    },
    setOpenAccordions: (state, action: PayloadAction<ExpandedIndex>) => {
      state.openAccordions = action.payload;
    },
  },
});

export const {
  setShouldDisplayInProgress,
  setIsProcessing,
  addLogEntry,
  setShouldShowLogViewer,
  setIsConnected,
  setSocketId,
  setShouldConfirmOnDelete,
  setOpenAccordions,
  setSystemStatus,
  setCurrentStatus,
} = systemSlice.actions;

export default systemSlice.reducer;
