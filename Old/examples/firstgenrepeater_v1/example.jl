struct RegisterTrait{T}
    trait::T
end

function RegisterTrait(trait::Any)
    trait_type = typeof(trait)

    # Type1 = Int32;
    # Type2 = String;
    # Type3 = Float64;

    is_valid_type = false;

    # Suppose that trait is expected to be of type Type1, Type2 or Type3, where 
    # these types can be very different from each other.
    if @isdefined Type1
        if trait_type == Type1
            is_valid_type = true
        end
    end

    if @isdefined Type2
        if trait_type == Type2
            is_valid_type = true
        end
    end

    if @isdefined Type3
        if trait_type == Type3
            is_valid_type = true
        end
    end

    if !is_valid_type
        @error "Unsupported trait type: $trait_type"
    end

    return RegisterTrait{trait_type}(trait)
end
