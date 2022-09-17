import { Grid, GridItem } from '@chakra-ui/react';
import CurrentImageDisplay from './features/gallery/CurrentImageDisplay';
import LogViewer from './features/system/LogViewer';
import PromptInput from './features/sd/PromptInput';
import ProgressBar from './features/header/ProgressBar';
import { useEffect } from 'react';
import { useAppDispatch } from './app/hooks';
import { requestAllImages } from './app/socketio';
import ProcessButtons from './features/sd/ProcessButtons';
import ImageGallery from './features/gallery/ImageGallery';
import SiteHeader from './features/header/SiteHeader';
import OptionsAccordion from './features/sd/OptionsAccordion';

const App = () => {
  const dispatch = useAppDispatch();

  // Load images from the gallery once
  useEffect(() => {
    dispatch(requestAllImages());
  }, [dispatch]);

  return (
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
  );
};

export default App;
