import {
    Slider,
    SliderTrack,
    SliderFilledTrack,
    SliderThumb,
    FormControl,
    FormLabel,
    Text,
    Flex,
    SliderProps,
} from '@chakra-ui/react';

interface Props extends SliderProps {
    label: string;
    value: number;
    fontSize?: number | string;
}

const SDSlider = ({
    label,
    value,
    fontSize = 'sm',
    onChange,
    ...rest
}: Props) => {
    return (
        <FormControl>
            <Flex gap={2}>
                <FormLabel marginInlineEnd={0} marginBottom={1}>
                    <Text fontSize={fontSize} whiteSpace='nowrap'>
                        {label}
                    </Text>
                </FormLabel>
                <Slider
                    aria-label={label}
                    focusThumbOnChange={true}
                    value={value}
                    onChange={onChange}
                    {...rest}
                >
                    <SliderTrack>
                        <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                </Slider>
            </Flex>
        </FormControl>
    );
};

export default SDSlider;
