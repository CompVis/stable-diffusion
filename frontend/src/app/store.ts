import { combineReducers, configureStore } from '@reduxjs/toolkit';
import { useDispatch, useSelector } from 'react-redux';
import type { TypedUseSelectorHook } from 'react-redux';

import { persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage for web

import optionsReducer from '../features/options/optionsSlice';
import galleryReducer from '../features/gallery/gallerySlice';
import systemReducer from '../features/system/systemSlice';
import { socketioMiddleware } from './socketio/middleware';

/**
 * redux-persist provides an easy and reliable way to persist state across reloads.
 *
 * While we definitely want generation parameters to be persisted, there are a number
 * of things we do *not* want to be persisted across reloads:
 *   - Gallery/selected image (user may add/delete images from disk between page loads)
 *   - Connection/processing status
 *   - Availability of external libraries like ESRGAN/GFPGAN
 *
 * These can be blacklisted in redux-persist.
 *
 * The necesssary nested persistors with blacklists are configured below.
 *
 * TODO: Do we blacklist initialImagePath? If the image is deleted from disk we get an
 * ugly 404. But if we blacklist it, then this is a valuable parameter that is lost
 * on reload. Need to figure out a good way to handle this.
 */

const rootPersistConfig = {
  key: 'root',
  storage,
  blacklist: ['gallery', 'system'],
};

const systemPersistConfig = {
  key: 'system',
  storage,
  blacklist: [
    'isConnected',
    'isProcessing',
    'currentStep',
    'socketId',
    'isESRGANAvailable',
    'isGFPGANAvailable',
    'currentStep',
    'totalSteps',
    'currentIteration',
    'totalIterations',
    'currentStatus',
  ],
};

const reducers = combineReducers({
  options: optionsReducer,
  gallery: galleryReducer,
  system: persistReducer(systemPersistConfig, systemReducer),
});

const persistedReducer = persistReducer(rootPersistConfig, reducers);

// Continue with store setup
export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      // redux-persist sometimes needs to temporarily put a function in redux state, need to disable this check
      serializableCheck: false,
    }).concat(socketioMiddleware()),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
