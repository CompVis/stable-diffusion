import {
    Box,
    Flex,
    Icon,
    IconButton,
    Image,
    useColorModeValue,
} from '@chakra-ui/react';
import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { SDImage, setCurrentImage } from './gallerySlice';
import { FaCheck, FaCopy, FaSeedling, FaTrash } from 'react-icons/fa';
import DeleteImageModalButton from './DeleteImageModalButton';
import { memo, SyntheticEvent, useState } from 'react';
import { setAllParameters, setSeed } from '../sd/sdSlice';

interface HoverableImageProps {
    image: SDImage;
    isSelected: boolean;
}

const HoverableImage = memo(
    (props: HoverableImageProps) => {
        const [isHovered, setIsHovered] = useState<boolean>(false);
        const dispatch = useAppDispatch();

        const checkColor = useColorModeValue('green.600', 'green.300');
        const bgColor = useColorModeValue('gray.200', 'gray.700');
        const bgGradient = useColorModeValue(
            'radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0.7) 20%, rgba(0,0,0,0) 100%)',
            'radial-gradient(circle, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.7) 20%, rgba(0,0,0,0) 100%)'
        );

        const { image, isSelected } = props;
        const { url, uuid, metadata } = image;

        const handleMouseOver = () => setIsHovered(true);
        const handleMouseOut = () => setIsHovered(false);
        const handleClickSetAllParameters = (e: SyntheticEvent) => {
            e.stopPropagation();
            dispatch(setAllParameters(metadata));
        };
        const handleClickSetSeed = (e: SyntheticEvent) => {
            e.stopPropagation();
            dispatch(setSeed(image.metadata.seed!)); // component not rendered unless this exists
        };

        return (
            <Box position={'relative'} key={uuid}>
                <Image
                    width={120}
                    height={120}
                    objectFit='cover'
                    rounded={'md'}
                    src={url}
                    loading={'lazy'}
                    backgroundColor={bgColor}
                />
                <Flex
                    cursor={'pointer'}
                    position={'absolute'}
                    top={0}
                    left={0}
                    rounded={'md'}
                    width='100%'
                    height='100%'
                    alignItems={'center'}
                    justifyContent={'center'}
                    background={isSelected ? bgGradient : undefined}
                    onClick={() => dispatch(setCurrentImage(image))}
                    onMouseOver={handleMouseOver}
                    onMouseOut={handleMouseOut}
                >
                    {isSelected && (
                        <Icon
                            fill={checkColor}
                            width={'50%'}
                            height={'50%'}
                            as={FaCheck}
                        />
                    )}
                    {isHovered && (
                        <Flex
                            direction={'column'}
                            gap={1}
                            position={'absolute'}
                            top={1}
                            right={1}
                        >
                            <DeleteImageModalButton image={image}>
                                <IconButton
                                    colorScheme='red'
                                    aria-label='Delete image'
                                    icon={<FaTrash />}
                                    size='xs'
                                    fontSize={15}
                                />
                            </DeleteImageModalButton>
                            <IconButton
                                aria-label='Use all parameters'
                                colorScheme={'blue'}
                                icon={<FaCopy />}
                                size='xs'
                                fontSize={15}
                                onClickCapture={handleClickSetAllParameters}
                            />
                            {image.metadata.seed && (
                                <IconButton
                                    aria-label='Use seed'
                                    colorScheme={'blue'}
                                    icon={<FaSeedling />}
                                    size='xs'
                                    fontSize={16}
                                    onClickCapture={handleClickSetSeed}
                                />
                            )}
                        </Flex>
                    )}
                </Flex>
            </Box>
        );
    },
    (prev, next) =>
        prev.image.uuid === next.image.uuid &&
        prev.isSelected === next.isSelected
);

const ImageRoll = () => {
    const { images, currentImageUuid } = useAppSelector(
        (state: RootState) => state.gallery
    );

    return (
        <Flex gap={2} wrap='wrap' pb={2}>
            {[...images].reverse().map((image) => {
                const { uuid } = image;
                const isSelected = currentImageUuid === uuid;
                return (
                    <HoverableImage
                        key={uuid}
                        image={image}
                        isSelected={isSelected}
                    />
                );
            })}
        </Flex>
    );
};

export default ImageRoll;
