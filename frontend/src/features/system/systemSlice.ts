import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { ExpandedIndex } from '@chakra-ui/react';
import * as InvokeAI from '../../app/invokeai';

export type LogLevel = 'info' | 'warning' | 'error';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
}

export interface Log {
  [index: number]: LogEntry;
}

export interface SystemState
  extends InvokeAI.SystemStatus,
    InvokeAI.SystemConfig {
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
  shouldDisplayGuides: boolean;
}

const initialSystemState = {
  isConnected: false,
  isProcessing: false,
  log: [],
  shouldShowLogViewer: false,
  shouldDisplayInProgress: false,
  shouldDisplayGuides: true,
  isGFPGANAvailable: true,
  isESRGANAvailable: true,
  socketId: '',
  shouldConfirmOnDelete: true,
  openAccordions: [0],
  currentStep: 0,
  totalSteps: 0,
  currentIteration: 0,
  totalIterations: 0,
  currentStatus: 'Disconnected',
  currentStatusHasSteps: false,
  model: '',
  model_id: '',
  model_hash: '',
  app_id: '',
  app_version: '',
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
    setSystemStatus: (state, action: PayloadAction<InvokeAI.SystemStatus>) => {
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
      state.isProcessing = false;
      state.currentStep = 0;
      state.totalSteps = 0;
      state.currentIteration = 0;
      state.totalIterations = 0;
      state.currentStatusHasSteps = false;
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
    setSystemConfig: (state, action: PayloadAction<InvokeAI.SystemConfig>) => {
      return { ...state, ...action.payload };
    },
    setShouldDisplayGuides: (state, action: PayloadAction<boolean>) => {
      state.shouldDisplayGuides = action.payload;
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
  setSystemConfig,
  setShouldDisplayGuides,
} = systemSlice.actions;

export default systemSlice.reducer;
