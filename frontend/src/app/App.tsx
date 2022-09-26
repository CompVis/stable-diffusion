import { Grid, GridItem } from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import CurrentImageDisplay from '../features/gallery/CurrentImageDisplay';
import ImageGallery from '../features/gallery/ImageGallery';
import ProgressBar from '../features/system/ProgressBar';
import SiteHeader from '../features/system/SiteHeader';
import OptionsAccordion from '../features/options/OptionsAccordion';
import ProcessButtons from '../features/options/ProcessButtons';
import PromptInput from '../features/options/PromptInput';
import LogViewer from '../features/system/LogViewer';
import Loading from '../Loading';
import { useAppDispatch } from './store';
import { requestSystemConfig } from './socketio/actions';

const App = () => {
  const dispatch = useAppDispatch();
  const [isReady, setIsReady] = useState<boolean>(false);

  useEffect(() => {
    dispatch(requestSystemConfig());
    setIsReady(true);
  }, [dispatch]);

  return isReady ? (
    <>
      <Grid
        width="100vw"
        height="100vh"
        templateAreas={`
                    "header header header header"
                    "progressBar progressBar progressBar progressBar"
                    "menu prompt processButtons imageRoll"
                    "menu currentImage currentImage imageRoll"`}
        gridTemplateRows={'36px 10px 100px auto'}
        gridTemplateColumns={'350px auto 100px 388px'}
        gap={2}
      >
        <GridItem area={'header'} pt={1}>
          <SiteHeader />
        </GridItem>
        <GridItem area={'progressBar'}>
          <ProgressBar />
        </GridItem>
        <GridItem pl="2" area={'menu'} overflowY="scroll">
          <OptionsAccordion />
        </GridItem>
        <GridItem area={'prompt'}>
          <PromptInput />
        </GridItem>
        <GridItem area={'processButtons'}>
          <ProcessButtons />
        </GridItem>
        <GridItem area={'currentImage'}>
          <CurrentImageDisplay />
        </GridItem>
        <GridItem pr="2" area={'imageRoll'} overflowY="scroll">
          <ImageGallery />
        </GridItem>
      </Grid>
      <LogViewer />
    </>
  ) : (
    <Loading />
  );
};

export default App;
