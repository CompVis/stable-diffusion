import { Grid, GridItem } from '@chakra-ui/react';
import CurrentImage from './features/gallery/CurrentImage';
import LogViewer from './features/system/LogViewer';
import PromptInput from './features/sd/PromptInput';
import ProgressBar from './features/header/ProgressBar';
import { useEffect } from 'react';
import { useAppDispatch } from './app/hooks';
import { requestAllImages } from './app/socketio';
import ProcessButtons from './features/sd/ProcessButtons';
import ImageRoll from './features/gallery/ImageRoll';
import SiteHeader from './features/header/SiteHeader';
import OptionsAccordion from './features/sd/OptionsAccordion';

const App = () => {
    const dispatch = useAppDispatch();
    useEffect(() => {
        dispatch(requestAllImages());
    }, [dispatch]);
    return (
        <>
            <Grid
                width='100vw'
                height='100vh'
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
                <GridItem pl='2' area={'menu'} overflowY='scroll'>
                    <OptionsAccordion />
                </GridItem>
                <GridItem area={'prompt'}>
                    <PromptInput />
                </GridItem>
                <GridItem area={'processButtons'}>
                    <ProcessButtons />
                </GridItem>
                <GridItem area={'currentImage'}>
                    <CurrentImage />
                </GridItem>
                <GridItem pr='2' area={'imageRoll'} overflowY='scroll'>
                    <ImageRoll />
                </GridItem>
            </Grid>
            <LogViewer />
        </>
    );
};

export default App;
