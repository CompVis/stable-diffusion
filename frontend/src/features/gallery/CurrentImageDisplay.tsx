import { Center, Flex, Image, Text, useColorModeValue } from '@chakra-ui/react';
import { useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { useState } from 'react';
import ImageMetadataViewer from './ImageMetadataViewer';
import CurrentImageButtons from './CurrentImageButtons';

// TODO: With CSS Grid I had a hard time centering the image in a grid item. This is needed for that.
const height = 'calc(100vh - 238px)';

/**
 * Displays the current image if there is one, plus associated actions.
 */
const CurrentImageDisplay = () => {
  const { currentImage, intermediateImage } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const bgColor = useColorModeValue(
    'rgba(255, 255, 255, 0.85)',
    'rgba(0, 0, 0, 0.8)'
  );

  const [shouldShowImageDetails, setShouldShowImageDetails] =
    useState<boolean>(false);

  const imageToDisplay = intermediateImage || currentImage;

  return imageToDisplay ? (
    <Flex direction={'column'} borderWidth={1} rounded={'md'} p={2} gap={2}>
      <CurrentImageButtons
        image={imageToDisplay}
        shouldShowImageDetails={shouldShowImageDetails}
        setShouldShowImageDetails={setShouldShowImageDetails}
      />
      <Center height={height} position={'relative'}>
        <Image
          src={imageToDisplay.url}
          fit="contain"
          maxWidth={'100%'}
          maxHeight={'100%'}
        />
        {shouldShowImageDetails && (
          <Flex
            width={'100%'}
            height={'100%'}
            position={'absolute'}
            top={0}
            left={0}
            p={3}
            boxSizing="border-box"
            backgroundColor={bgColor}
            overflow="scroll"
          >
            <ImageMetadataViewer image={imageToDisplay} />
          </Flex>
        )}
      </Center>
    </Flex>
  ) : (
    <Center height={'100%'} position={'relative'}>
      <Text size={'xl'}>No image selected</Text>
    </Center>
  );
};

export default CurrentImageDisplay;
