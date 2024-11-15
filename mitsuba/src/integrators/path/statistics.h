#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>

#include <string>
#include <vector>

MTS_NAMESPACE_BEGIN

template<template<typename Data> class Entry>
struct Stats {
    template<typename T, int Count>
    struct Stack {
#ifdef MARS_INCLUDE_AOVS
        std::unique_ptr<Entry<T>> first;
        std::unique_ptr<Entry<T>> entries[Count];

        int minDepth = -1;

        Stack(const std::string &name) {
            first.reset(new Entry<T>(name));
            for (int i = 0; i < Count; ++i) {
                entries[i].reset(new Entry<T>(name + "." + std::to_string(i)));
            }
        }

        void reset() {
            minDepth = -1;
            first->reset();
            for (int i = 0; i < Count; ++i)
                entries[i]->reset();
        }

        void add(int depth, const T &value, Float weight = 1) {
            SAssert(depth >= 0);

            if constexpr (Count > 0) {
                if (depth < Count)
                    entries[depth]->add(value, weight);
            }

            if (depth <= minDepth || minDepth == -1) {
                if (depth < minDepth)
                    first->reset();
                minDepth = depth;
                first->add(value, weight);
            }
        }
#else
    Stack(const std::string &name) {}
    void reset() {}
    void add(int depth, const T &value, Float weight = 1) {}
#endif
    };

    /// do not change or re-order these! they are used for denoising and other stuff.
    Stack<Spectrum, 0> albedo { "px.albedo" };
    Stack<Spectrum, 0> normal { "px.normal" };
    Entry<Spectrum>    relvar { "px.relvar" };

    /// feel free to change everything below this line though!
    Entry<Spectrum>    pixelEstimate { "px.estimate"     };
    Entry<Float>       pixelCost     { "px.cost"         };
    Entry<Float>       avgPathLength { "paths.length"    };

    Stack<Float, 0>    primaryCost        { "est.pcost"   };
    Stack<Float, 0>    primaryCostBsdf    { "est.pcostB"  };
    Stack<Float, 0>    primaryCostGuid    { "est.pcostG"  };
    Stack<Float, 0>    primaryCostNee     { "est.pcostN"  };
    Stack<Spectrum, 0> primary2ndMom      { "est.p2ndMom" };
    Stack<Spectrum, 0> primary2ndMomBsdf  { "est.p2ndMomB"};
    Stack<Spectrum, 0> primary2ndMomGuid  { "est.p2ndMomG"};
    Stack<Spectrum, 0> primary2ndMomNee   { "est.p2ndMomN"};
    Stack<Spectrum, 0> primaryVar         { "est.pVar"    };
    Stack<Spectrum, 0> primaryVarBsdf     { "est.pVarB"   };
    Stack<Spectrum, 0> primaryVarGuid     { "est.pVarG"   };
    Stack<Spectrum, 0> primaryVarNee      { "est.pVarN"   };
    Stack<Spectrum, 3> irradiance         { "est.irr"     };
    Stack<Spectrum, 0> indirectIrradiance { "est.irrI"    };
    Stack<Float, 0>    irradianceWeight   { "est.irrW"    };
    Stack<Spectrum, 3> liEstimate         { "est.Li"      };

    Stack<Float, 4>    splittingFactorB   { "d.rrsB"      };
    Stack<Float, 4>    splittingFactorG   { "d.rrsG"      };
    Stack<Float, 4>    splittingFactorN   { "d.rrsN"      };
    Stack<Float, 0>    bsdfFraction       { "d.bsdfFrac"  };
    Stack<Float, 0>    roughness          { "d.roughness" };

    void reset() {
        albedo.reset();
        normal.reset();
        // don't reset pixelEstimate, it's still needed.

        relvar.reset();
        pixelCost.reset();
        avgPathLength.reset();

        primaryCost.reset();
        primaryCostBsdf.reset();
        primaryCostGuid.reset();
        primaryCostNee.reset();

        primary2ndMom.reset();
        primary2ndMomBsdf.reset();
        primary2ndMomGuid.reset();
        primary2ndMomNee.reset();

        primaryVar.reset();
        primaryVarBsdf.reset();
        primaryVarGuid.reset();
        primaryVarNee.reset();

        irradiance.reset();
        indirectIrradiance.reset();
        irradianceWeight.reset();
        liEstimate.reset();

        splittingFactorB.reset();
        splittingFactorG.reset();
        splittingFactorN.reset();
        bsdfFraction.reset();
        roughness.reset();
    }
};

template<typename T>
struct FormatDescriptor {};

template<>
struct FormatDescriptor<Float> {
    int numComponents = 1;
    Bitmap::EPixelFormat pixelFormat = Bitmap::EPixelFormat::ELuminance;
    std::string pixelName = "luminance";
};

template<>
struct FormatDescriptor<Spectrum> {
    int numComponents = SPECTRUM_SAMPLES;
    Bitmap::EPixelFormat pixelFormat = Bitmap::EPixelFormat::ESpectrum;
    std::string pixelName = "rgb";
};

struct StatsImageBlockCache {
    thread_local static StatsImageBlockCache *instance;
    StatsImageBlockCache(std::function<ImageBlock *()> createImage)
    : createImage(createImage) {
        instance = this;
    }

    std::function<ImageBlock *()> createImage;
    mutable std::vector<ref<ImageBlock>> blocks;
};

template<typename T>
struct StatsImageBlockEntry {
    StatsImageBlockEntry(const std::string &) {
        image = StatsImageBlockCache::instance->createImage();
        image->setWarn(false); // some statistics can be negative
        StatsImageBlockCache::instance->blocks.push_back(image);
    }

    ImageBlock *image;

    void add(const T &, Float) {}
};

struct StatsImageBlocks : StatsImageBlockCache, Stats<StatsImageBlockEntry> {
    StatsImageBlocks(std::function<ImageBlock *()> createImage)
    : StatsImageBlockCache(createImage) {}

    void clear() {
        for (auto &block : blocks)
            block->clear();
    }

    void put(StatsImageBlocks &other) const {
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i]->put(other.blocks[i]);
        }
    }

    std::vector<Bitmap *> getBitmaps() {
        std::vector<Bitmap *> result;
        for (auto &block : blocks)
            result.push_back(block->getBitmap());
        return result;
    }
};

struct StatsDescriptorCache {
    thread_local static StatsDescriptorCache *instance;
    StatsDescriptorCache() {
        instance = this;
    }

    std::string names = "color", types = "rgb";

    int size = 1;
    int components = SPECTRUM_SAMPLES;
};

template<typename T>
struct StatsDescriptorEntry {
    StatsDescriptorEntry(const std::string &name) {
        auto &cache = *StatsDescriptorCache::instance;

        cache.names += ", " + name;

        FormatDescriptor<T> fmt;
        cache.components += fmt.numComponents;
        cache.types += ", " + fmt.pixelName;

        cache.size += 1;
    }

    void add(const T &, Float) {}
};

struct StatsDescriptor : StatsDescriptorCache, Stats<StatsDescriptorEntry> {
};

struct StatsValuesCache {
    thread_local static StatsValuesCache *instance;
    StatsValuesCache() {
        instance = this;
    }

    std::vector<std::function<void (ImageBlock *, const Point2 &, Float)>> putters;
};

template<typename T>
struct StatsValueEntry {
    StatsValueEntry(const std::string &) {
        StatsValuesCache::instance->putters.push_back([&](ImageBlock *block, const Point2 &samplePos, Float alpha) {
            if (weight > 0.f)
                alpha /= weight;
            block->put(samplePos, Spectrum { value }, alpha);
        });
    }

    T value { 0.f };
    Float weight = 0.f;

    void reset() {
        value = T { 0.f };
        weight = 0.f;
    }

    void increment() {
        value++;
    }

    void add(const T &v, Float w = 1) {
        value += v;
        weight += w;
    }
};

struct StatsValues : StatsValuesCache, Stats<StatsValueEntry> {
    void put(StatsImageBlocks &other, const Point2 &samplePos, Float alpha) {
        for (size_t i = 0; i < putters.size(); ++i)
            putters[i](other.blocks[i], samplePos, alpha);
    }
};

MTS_NAMESPACE_END
